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
import time
from datautils_block import BlockTrainDataset
from torch.utils.data import DataLoader
import shutil
import os

def update_dataloader(layer, dataset, dev, attention_mask, position_ids, offload=False):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for index1, inps in enumerate(dataset):
                inps = inps.to(dev).unsqueeze(0)
                new_data = layer(inps, attention_mask=attention_mask,position_ids=position_ids)[0].to('cpu')
                dataset.update_data(index1,new_data)

                    
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
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # move embedding layer and first layer to target device
    # only suppress llama models now
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    is_llama = True     
    
    
    layers[0] = layers[0].to(dev)
    dtype = torch.float16

    # path to save intermediate data
    flag = time.time()
    fp_train_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_train'
    fp_val_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_val'
    quant_train_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_train'
    quant_val_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_val'
    for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
        if os.path.exists(path):
            shutil.rmtree(path)

    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                model.config.hidden_size, dtype, cache_path=fp_train_cache_path,off_load_to_disk=args.off_load_to_disk)
    fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                model.config.hidden_size, dtype, cache_path=fp_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    fp_train_inps_loader = DataLoader(fp_train_inps, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,prefetch_factor=args.prefetch_factor)
    fp_val_inps_loader = DataLoader(fp_val_inps, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,prefetch_factor=args.prefetch_factor)
    
    # catch the first layer input
    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            if cache["i"]<=args.train_size-1:
                fp_train_inps.update_data(cache["i"], inp.squeeze(0).to('cpu'))
            else:
                fp_val_inps.update_data(cache["i"]-args.train_size, inp.squeeze(0).to('cpu'))
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError
    


    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama


    with torch.no_grad():
        for batch in (trainloader+valloader):
            if cache["i"] >= args.train_size + args.val_size:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    if args.off_load_to_disk:
        # copy quant input from fp input, they are same in first layer
        shutil.copytree(fp_train_cache_path, quant_train_cache_path)
        shutil.copytree(fp_val_cache_path, quant_val_cache_path)
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    else:
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
        for index,data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)
        for index,data in enumerate(fp_val_inps):
            quant_val_inps.update_data(index, data)
            
    quant_train_inps_loader = DataLoader(quant_train_inps, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,prefetch_factor=args.prefetch_factor)
    quant_val_inps_loader = DataLoader(quant_val_inps, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,prefetch_factor=args.prefetch_factor)

    
    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None
    
    

    for block_index in range(len(layers)):
        logger.info(f"=== Start quantize blocks {block_index}===")
        layer = layers[block_index].to(dev)
        qlayer = copy.deepcopy(layer)
        # replace torch.nn.Linear with QuantLinear for QAT
        for name, module in qlayer.named_modules():
            if isinstance(module,torch.nn.Linear):
                quantlinear = int_linear_fake.QuantLinear(module, args.wbits, args.group_size)
                set_op_by_name(qlayer, name, quantlinear)  
                del module  
        
        
        # obtain output of full-precision model
        qlayer.to(dev)
        # deactivate quantization for obtaining ground truth
        set_quant_state(qlayer,weight_quant=False)
        if args.epochs > 0:
            update_dataloader(qlayer,fp_train_inps,dev,attention_mask,position_ids)
            update_dataloader(qlayer,fp_val_inps,dev,attention_mask,position_ids)

        # activate quantization
        set_quant_state(qlayer,weight_quant=True)  
        
        total_training_iteration = args.epochs * args.train_size / args.batch_size
        
        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # fp32 is required for AMP training
            # create optimizer
            param = []
            assert args.quant_lr > 0 or args.weight_lr > 0
            param_group_index = 0
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
                loss_list = []
                norm_list = []
                start_time = time.time()
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps_loader, fp_train_inps_loader)):    
                    # obtain output of quantization model
                    with torch.cuda.amp.autocast():
                        input = quant_inps.to(dev)
                        label = fp_inps.to(dev)
                        quant_out = qlayer(input, attention_mask=attention_mask_batch,position_ids=position_ids)[0]
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

                val_loss_list = []
                for index, (quant_inps,fp_inps) in enumerate(zip(quant_val_inps_loader, fp_val_inps_loader)):  
                    # obtain output of quantization model
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            input = quant_inps.to(dev)
                            label = fp_inps.to(dev)
                            quant_out = qlayer(input, attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                            reconstruction_loss = loss_func(label, quant_out)
                    val_loss_list.append(reconstruction_loss.cpu())
                 
                train_mean_num = min(len(loss_list),64) # calculate the average training loss of last train_mean_num samples
                loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"blocks {block_index} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} time {time.time()-start_time} ")
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                else:
                    early_stop_flag += 1
                    if args.early_stop > 0 and early_stop_flag >=args.early_stop:
                        break
            optimizer.zero_grad()
            del optimizer

        # real smooth and quantization
        qlayer.half()
        # directly replace the weight with fake quantization
        quant_inplace(qlayer)
        set_quant_state(qlayer,weight_quant=False)  # weight has been quantized inplace
        if args.epochs>0:
            # update inputs of quantization model
            update_dataloader(qlayer,quant_train_inps,dev,attention_mask,position_ids)
            update_dataloader(qlayer,quant_val_inps,dev,attention_mask,position_ids)
        
        # move to cpu
        layers[block_index] = qlayer.to("cpu")

        # pack quantized weights, note that this process is slow on poor CPU or busy CPU
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
    for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
        if os.path.exists(path):
            shutil.rmtree(path)

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

