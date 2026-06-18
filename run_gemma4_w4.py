"""Reproducible Gemma-4 Block-AP run: measure PPL before and after w4 quant.

Two eval backends, same math (BOS-prefixed deterministic windows, loss over seqlen):
  - eval_ppl      : whole model forward. Simple; use on CPU (full 5.1B won't fit a 10GB GPU).
  - eval_ppl_gpu  : block-walk — only pre-layers + one block on GPU at a time, so the model
                    evals on a 10GB card and ~10x faster than CPU. Reproduces the model's own
                    logits (final norm + lm_head + softcap).
PPL is over a fixed slice of the test set (seed-free, deterministic) so fp16 vs quant is a
clean apples-to-apples comparison; raise --eval_seqs for a tighter estimate.
"""
import argparse, types, random
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import utils
from datautils_block import get_loaders
from quantize.block_ap import block_ap
from quantize.model_adapters import get_adapter

MID = "google/gemma-4-E2B-it"


@torch.no_grad()
def eval_ppl(model, tokenizer, dataset, seqlen, n_seqs):
    """Deterministic PPL over the first n_seqs non-overlapping windows of the test set.

    Each window is prefixed with BOS: Gemma's tokenizer has add_bos_token=False and the
    model is very BOS-sensitive — without it PPL is ~1000x inflated. We measure loss only
    on the real tokens (the BOS-predicted first token is excluded), so PPL stays over seqlen.
    """
    bos = tokenizer.bos_token_id
    testenc = get_loaders(dataset, tokenizer, seqlen=seqlen, test_only=True)
    testenc = getattr(testenc, "input_ids", testenc)  # wikitext returns an encoding, c4 a tensor
    n_seqs = min(n_seqs, testenc.numel() // seqlen)
    nlls = []
    for i in range(n_seqs):
        batch = testenc[:, i * seqlen:(i + 1) * seqlen]
        batch = torch.cat([torch.tensor([[bos]]), batch], dim=1)  # prepend BOS
        logits = model(batch).logits.float()
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = batch[:, 1:].reshape(-1).to(shift_logits.device)
        nlls.append(torch.nn.functional.cross_entropy(shift_logits, shift_labels) * seqlen)
    return torch.exp(torch.stack(nlls).sum() / (n_seqs * seqlen)).item()


@torch.no_grad()
def eval_ppl_gpu(model, tokenizer, dataset, seqlen, n_seqs, dev):
    """Same PPL as eval_ppl, but walks the model one block at a time on `dev` so the full
    5.1B never sits on the GPU at once (pre-layers + lm_head + a single block ~7GB, fits 10GB).
    Reproduces the model's logits exactly: final norm -> lm_head -> tanh logit-softcap."""
    adapter = get_adapter(model)
    adapter._ple_cache = {}
    root, layers = adapter._root, adapter.layers
    bos = tokenizer.bos_token_id
    cap = model.config.get_text_config().final_logit_softcapping
    need_ple = adapter.needs_per_layer_input()
    testenc = get_loaders(dataset, tokenizer, seqlen=seqlen, test_only=True)
    testenc = getattr(testenc, "input_ids", testenc)
    n_seqs = min(n_seqs, testenc.numel() // seqlen)

    adapter.move_pre_layers(dev)
    norm, lm_head = adapter.final_norm().to(dev), model.lm_head.to(dev)

    def layer0_input(ids):
        # capture the post-embedding hidden state that layer 0 receives (the model applies
        # embedding scaling there) by short-circuiting the real forward at layer 0.
        box, orig = {}, layers[0].forward
        def grab(inp, *a, **kw):
            box["h"] = inp
            raise StopIteration
        layers[0].forward = grab
        try:
            root(ids, use_cache=False)
        except StopIteration:
            pass
        finally:
            layers[0].forward = orig
        return box["h"]

    nlls = []
    for i in range(n_seqs):
        batch = testenc[:, i * seqlen:(i + 1) * seqlen]
        ids = torch.cat([torch.tensor([[bos]]), batch], dim=1).to(dev)
        adapter._ple_cache.clear()  # ids is transient per window; avoid id() reuse collisions
        if i == 0:
            adapter.capture_static_kwargs(lambda: adapter.forward_for_capture(ids))
        hidden = layer0_input(ids)
        shared = {}
        for bi, layer in enumerate(layers):
            layer.to(dev)
            ple = adapter.per_layer_input(bi, ids) if need_ple else None
            produce = adapter.is_kv_producer(bi) is not None
            skv = shared if adapter.is_kv_consumer(bi) else None
            hidden, produced = adapter.run_block(bi, layer, hidden, per_layer_input=ple,
                                                 shared_kv=skv, produce_shared_kv=produce)
            if produced:
                shared.update(produced)
            layer.to("cpu")
        logits = lm_head(norm(hidden)).float()
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        nlls.append(torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:]) * seqlen)

    adapter.move_pre_layers("cpu")
    adapter.final_norm().to("cpu"); model.lm_head.to("cpu")
    torch.cuda.empty_cache()
    return torch.exp(torch.stack(nlls).sum() / (n_seqs * seqlen)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wbits", type=int, default=4)
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--calib_dataset", default="c4")
    p.add_argument("--eval_dataset", default="c4")
    p.add_argument("--train_size", type=int, default=256)
    p.add_argument("--val_size", type=int, default=16)
    p.add_argument("--training_seqlen", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--eval_seqlen", type=int, default=2048)
    p.add_argument("--eval_seqs", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cuda", "cpu"], help="block_ap compute device")
    p.add_argument("--eval_device", default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cuda", "cpu"], help="eval backend: cuda=block-walk (fast, low-mem), cpu=whole model")
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args()

    def run_eval(model):
        if args.eval_device == "cuda":
            return eval_ppl_gpu(model, tok, args.eval_dataset, args.eval_seqlen, args.eval_seqs,
                                torch.device("cuda"))
        return eval_ppl(model, tok, args.eval_dataset, args.eval_seqlen, args.eval_seqs)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    Path("./log").mkdir(exist_ok=True); Path("./cache").mkdir(exist_ok=True)
    logger = utils.create_logger(Path("./log"))
    logger.info(args)

    tok = AutoTokenizer.from_pretrained(MID)
    # bf16, not fp16: Gemma's activations overflow fp16 (gives garbage PPL ~1e4).
    model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cpu")
    for prm in model.parameters():
        prm.requires_grad = False

    ppl_fp16 = run_eval(model)
    logger.info(f"fp16 {args.eval_dataset} PPL ({args.eval_seqs} seqs): {ppl_fp16:.3f}")

    qargs = types.SimpleNamespace(
        cache_dir="./cache", calib_dataset=args.calib_dataset, train_size=args.train_size,
        val_size=args.val_size, training_seqlen=args.training_seqlen, batch_size=args.batch_size,
        epochs=args.epochs, wbits=args.wbits, group_size=args.group_size, quant_lr=1e-4,
        weight_lr=1e-5, min_lr_factor=20, wd=0, clip_grad=0.3, early_stop=0,
        off_load_to_disk=False, real_quant=False, net=f"gemma4-w{args.wbits}",
        device=args.device,
    )
    train, val = get_loaders(args.calib_dataset, tok, args.train_size, args.val_size,
                             seed=args.seed, seqlen=args.training_seqlen)
    block_ap(model, qargs, train, val, logger)

    ppl_q = run_eval(model)
    logger.info(f"w{args.wbits}g{args.group_size} {args.eval_dataset} PPL ({args.eval_seqs} seqs): {ppl_q:.3f}")
    label = "RTN (no train)" if args.epochs == 0 else f"trained {args.epochs}ep"
    logger.info(f"RESULT [{label}] fp16={ppl_fp16:.3f}  w{args.wbits}g{args.group_size}={ppl_q:.3f}  "
                f"delta=+{ppl_q - ppl_fp16:.3f}")


if __name__ == "__main__":
    main()
