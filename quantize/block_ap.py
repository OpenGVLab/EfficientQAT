import torch
import torch.nn as nn
import torch.nn.functional as F
import quantize.int_linear_fake as int_linear_fake
import quantize.int_linear_real as int_linear_real
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import math
import utils
import pdb
import gc
from quantize.utils import (
    quant_parameters,weight_parameters,trainable_parameters,
    set_quant_state,quant_inplace,set_quant_parameters,
    set_weight_parameters,trainable_parameters_num,get_named_linears,set_op_by_name)
from quantize.model_adapters import get_adapter
import time
from datautils_block import BlockTrainDataset
from torch.utils.data import DataLoader
import shutil
import os


from contextlib import nullcontext


def _autocast(dev, dtype):
    """autocast on CUDA in the model's dtype (bf16 for Gemma, fp16 for Llama); a no-op
    on CPU (blocks run in their native dtype there)."""
    return torch.autocast(device_type="cuda", dtype=dtype) if dev.type == "cuda" else nullcontext()


def _move(obj, dev):
    """Recursively move tensors in nested tuples/dicts to a device."""
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.to(dev)
    if isinstance(obj, dict):
        return {k: _move(v, dev) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(_move(v, dev) for v in obj)
    return obj


def run_block_over_dataset(adapter, block_index, layer, dataset, dev, dtype, id_batches,
                           shared_in=None, shared_out=None):
    """Run `layer` (block_index) over every batch in `dataset`, writing its output
    back in place. Threads per-batch PLE and shared_kv via the adapter; if the block
    is a KV producer, captures its shared_kv into `shared_out`."""
    produce = adapter.is_kv_producer(block_index) is not None
    is_consumer = adapter.is_kv_consumer(block_index)
    need_ple = adapter.needs_per_layer_input()
    with torch.no_grad():
        with _autocast(dev, dtype):
            for index, inps in enumerate(dataset):
                inps = inps.to(dev)
                if len(inps.shape) == 2:
                    inps = inps.unsqueeze(0)
                ple = adapter.per_layer_input(block_index, id_batches[index]).to(dev) if need_ple else None
                skv = _move(shared_in[index], dev) if (is_consumer and shared_in is not None) else None
                out, produced = adapter.run_block(block_index, layer, inps,
                                                  per_layer_input=ple, shared_kv=skv,
                                                  produce_shared_kv=produce)
                dataset.update_data(index, out.to('cpu'))
                if produce and shared_out is not None:
                    prev = shared_out[index] or {}
                    prev.update(_move(dict(produced), 'cpu'))
                    shared_out[index] = prev


def block_ap(
    model,
    args,
    trainloader,
    valloader,
    logger=None,
):
    logger.info("Starting ...")
    if args.off_load_to_disk:
        logger.info("offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")

    # device: explicit args.device wins, else CUDA if present. Lets you validate the CPU
    # path on a GPU box, then run the full job on GPU unchanged.
    dev = torch.device(getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu"))
    adapter = get_adapter(model)
    logger.info(f"using block-AP adapter: {adapter.name}")
    # use_cache may live on the text sub-config (multimodal wrappers like Gemma-4)
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    use_cache = getattr(cfg, "use_cache", None)
    cfg.use_cache = False

    # step 1: move embedding/pre-layer modules and first layer to device
    layers = adapter.layers
    adapter.move_pre_layers(dev)
    layers[0] = layers[0].to(dev)
    # follow the model's own dtype: Gemma loads bf16 (fp16 overflows its activations),
    # Llama loads fp16. Hardcoding fp16 here silently breaks Gemma.
    dtype = next(model.parameters()).dtype
    hidden_size = adapter.hidden_size

    # step 2: init dataset
    flag = time.time()
    if args.off_load_to_disk:
        fp_train_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_train'
        fp_val_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_val'
        quant_train_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_train'
        quant_val_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_val'
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
    else:
        fp_train_cache_path = None
        fp_val_cache_path = None
        quant_train_cache_path = None
        quant_val_cache_path = None
    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen,
                                hidden_size, args.batch_size, dtype, cache_path=fp_train_cache_path,off_load_to_disk=args.off_load_to_disk)
    fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen,
                                hidden_size, args.batch_size, dtype, cache_path=fp_val_cache_path,off_load_to_disk=args.off_load_to_disk)

    # step 2.1: keep input_ids per batch (needed by adapters that compute per-layer inputs, e.g. Gemma PLE)
    def build_id_batches(loader):
        n = len(loader) // args.batch_size
        return [torch.cat([loader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
                for i in range(n)]
    train_id_batches = build_id_batches(trainloader)
    val_id_batches = build_id_batches(valloader)

    # step 3: catch the input of the first layer
    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0

        def forward(self, inp, *args, **kwargs):
            self.dataset.update_data(self.index, inp.squeeze(0).to('cpu'))
            self.index += 1
            raise ValueError

    # step 3.1: catch the input of training set
    layers[0] = Catcher(layers[0],fp_train_inps)
    iters = len(trainloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    # step 3.2: catch the input of validation set
    layers[0] = Catcher(layers[0],fp_val_inps)
    iters = len(valloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([valloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    # step 3.3: capture per-layer static kwargs (attention masks, position ids/embeddings).
    # These are sample-independent at fixed seqlen. One identity-patched forward covers all layers.
    adapter.capture_static_kwargs(lambda: adapter.forward_for_capture(train_id_batches[0].to(dev)))
    if adapter.static[0]:
        logger.info(f"captured static block kwargs: {list(adapter.static[0].keys())}")

    # step 4: move embedding/pre-layer modules and first layer to cpu
    layers[0] = layers[0].cpu()
    adapter.move_pre_layers('cpu')
    torch.cuda.empty_cache()

    # step 5: copy fp input as the quant input, they are same at the first layer
    if args.off_load_to_disk:
        shutil.copytree(fp_train_cache_path, quant_train_cache_path)
        shutil.copytree(fp_val_cache_path, quant_val_cache_path)
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen,
                                    hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen,
                                    hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    else:
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen,
                                    hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen,
                                    hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
        for index,data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)
        for index,data in enumerate(fp_val_inps):
            quant_val_inps.update_data(index, data)

    # step 5.1: per-sample shared_kv stores (KV-sharing models only). Filled at producer
    # blocks; read by consumer blocks. fp/quant kept separate (mirror the input datasets).
    # ponytail: in-memory lists; offload alongside off_load_to_disk if it ever OOMs CPU.
    shared_fp_train = [None]*len(train_id_batches)
    shared_fp_val = [None]*len(val_id_batches)
    shared_quant_train = [None]*len(train_id_batches)
    shared_quant_val = [None]*len(val_id_batches)

    # step 6: start training
    loss_func = torch.nn.MSELoss()
    for block_index in range(len(layers)):
        logger.info(f"=== Start quantize blocks {block_index}===")
        is_consumer = adapter.is_kv_consumer(block_index)
        need_ple = adapter.needs_per_layer_input()
        # step 6.1: replace torch.nn.Linear with QuantLinear for QAT
        layer = layers[block_index].to(dev)
        qlayer = copy.deepcopy(layer)
        for name, module in qlayer.named_modules():
            if isinstance(module,torch.nn.Linear):
                quantlinear = int_linear_fake.QuantLinear(module, args.wbits, args.group_size)
                set_op_by_name(qlayer, name, quantlinear)
                del module
        qlayer.to(dev)

        # step 6.2: obtain output of full-precision model for MSE (also fills fp shared_kv if producer)
        set_quant_state(qlayer,weight_quant=False) # deactivate quantization for obtaining ground truth
        if args.epochs > 0:
            run_block_over_dataset(adapter, block_index, qlayer, fp_train_inps, dev, dtype, train_id_batches,
                                   shared_in=shared_fp_train, shared_out=shared_fp_train)
            run_block_over_dataset(adapter, block_index, qlayer, fp_val_inps, dev, dtype, val_id_batches,
                                   shared_in=shared_fp_val, shared_out=shared_fp_val)
        set_quant_state(qlayer,weight_quant=True)  # activate quantization


        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # fp32 is required for AMP training
            # step 6.3: create optimizer and learning rate schedule
            param = []
            assert args.quant_lr > 0 or args.weight_lr > 0
            param_group_index = 0
            total_training_iteration = args.epochs * args.train_size / args.batch_size
            if args.quant_lr > 0:
                set_quant_parameters(qlayer,True)
                param.append({"params":quant_parameters(qlayer),"lr":args.quant_lr})
                empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.quant_lr)
                quant_scheduler = CosineAnnealingLR(empty_optimizer_1, T_max=total_training_iteration, eta_min=args.quant_lr/args.min_lr_factor)
                quant_index = param_group_index
                param_group_index += 1
            else:
                set_quant_parameters(qlayer,False)

            if args.weight_lr > 0:
                set_weight_parameters(qlayer,True)
                param.append({"params":weight_parameters(qlayer),"lr":args.weight_lr})
                empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
                weight_scheduler = CosineAnnealingLR(empty_optimizer_2, T_max=total_training_iteration, eta_min=args.weight_lr/args.min_lr_factor)
                weight_index = param_group_index
                param_group_index += 1
            else:
                set_weight_parameters(qlayer,False)
            optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            trainable_number = trainable_parameters_num(qlayer)
            print(f"trainable parameter number: {trainable_number/1e6}M")

            best_val_loss = 1e6
            early_stop_flag = 0
            for epoch in range(args.epochs):
                # step: 6.4 training
                loss_list = []
                norm_list = []
                start_time = time.time()
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps, fp_train_inps)):
                    # obtain output of quantization model
                    with _autocast(dev, dtype):
                        input = quant_inps.to(dev)
                        label = fp_inps.to(dev)
                        ple = adapter.per_layer_input(block_index, train_id_batches[index]).to(dev) if need_ple else None
                        skv = _move(shared_quant_train[index], dev) if is_consumer else None
                        quant_out, _ = adapter.run_block(block_index, qlayer, input,
                                                         per_layer_input=ple, shared_kv=skv)
                        reconstruction_loss = loss_func(label, quant_out)
                        loss =  reconstruction_loss

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                    loss_list.append(reconstruction_loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters=trainable_parameters(qlayer)).cpu()
                    norm_list.append(norm.data)

                    # adjust lr
                    if args.quant_lr > 0:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]
                    if args.weight_lr >0 :
                        weight_scheduler.step()
                        optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[0]

                # step 6.5: calculate validation loss
                val_loss_list = []
                for index, (quant_inps,fp_inps) in enumerate(zip(quant_val_inps, fp_val_inps)):
                    # obtain output of quantization model
                    with torch.no_grad():
                        with _autocast(dev, dtype):
                            input = quant_inps.to(dev)
                            label = fp_inps.to(dev)
                            ple = adapter.per_layer_input(block_index, val_id_batches[index]).to(dev) if need_ple else None
                            skv = _move(shared_quant_val[index], dev) if is_consumer else None
                            quant_out, _ = adapter.run_block(block_index, qlayer, input,
                                                             per_layer_input=ple, shared_kv=skv)
                            reconstruction_loss = loss_func(label, quant_out)
                    val_loss_list.append(reconstruction_loss.cpu())

                train_mean_num = min(len(loss_list),64) # calculate the average training loss of last train_mean_num samples
                loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                max_mem = torch.cuda.max_memory_allocated(dev) / 1024**2 if dev.type == "cuda" else 0
                logger.info(f"blocks {block_index} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {max_mem} time {time.time()-start_time} ")
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                else:
                    early_stop_flag += 1
                    if args.early_stop > 0 and early_stop_flag >=args.early_stop:
                        break
            optimizer.zero_grad()
            del optimizer

        # step 6.6: directly replace the weight with fake quantization
        qlayer.to(dtype)
        quant_inplace(qlayer)
        set_quant_state(qlayer,weight_quant=False)  # weight has been quantized inplace

        # step 6.7: update inputs of quantization model (also fills quant shared_kv if producer)
        if args.epochs>0:
            run_block_over_dataset(adapter, block_index, qlayer, quant_train_inps, dev, dtype, train_id_batches,
                                   shared_in=shared_quant_train, shared_out=shared_quant_train)
            run_block_over_dataset(adapter, block_index, qlayer, quant_val_inps, dev, dtype, val_id_batches,
                                   shared_in=shared_quant_val, shared_out=shared_quant_val)
        layers[block_index] = qlayer.to("cpu")

        # step 7: pack quantized weights into low-bits format, note that this process is slow on poor CPU or busy CPU
        if args.real_quant:
            named_linears = get_named_linears(qlayer, int_linear_fake.QuantLinear)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scale.clamp(1e-4,1e4).detach()
                zeros = module.weight_quantizer.zero_point.detach().cuda().round().cpu()
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1).transpose(0,1).contiguous()
                zeros = zeros.view(dim0,-1).transpose(0,1).contiguous()
                q_linear = int_linear_real.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                set_op_by_name(qlayer, name, q_linear)
                logger.info(f"pack quantized {name} finished")
                del module
        del layer
        torch.cuda.empty_cache()

    # delete cached dataset
    if args.off_load_to_disk:
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    torch.cuda.empty_cache()
    gc.collect()
    if use_cache is not None:
        cfg.use_cache = use_cache
    return model
