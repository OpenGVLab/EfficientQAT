# Gemma-4 Block-AP quantization

Block-wise quantization-aware training (Block-AP) for `google/gemma-4-E2B-it`
(Gemma-3n / MatFormer lineage: PLE, KV-sharing, alternating sliding/global
attention, multimodal wrapper). Vision/audio encoders are left in fp16; only the
text decoder is quantized.

Model-specific logic lives in `quantize/model_adapters.py` (`Gemma4Adapter`);
`quantize/block_ap.py` is model-agnostic. To support another model, add an adapter
and register it in `get_adapter` — no changes to `block_ap.py`.

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .                       # deps from pyproject.toml
huggingface-cli download google/gemma-4-E2B-it   # ~10GB, needs HF access to the gated repo
```

## Run

End-to-end: fp16 baseline PPL → Block-AP w4 → quantized PPL → delta.

```bash
# Full run on GPU (safe for a 10GB card + ~21GB RAM)
python run_gemma4_w4.py --device cuda --eval_device cuda \
  --calib_dataset c4 --eval_dataset c4 \
  --train_size 256 --val_size 16 --training_seqlen 1024 --batch_size 2 \
  --epochs 2 --eval_seqlen 2048 --eval_seqs 40
```

Output (in `./log/`):
```
RESULT [trained 2ep] fp16=<a>  w4g128=<b>  delta=+<b-a>
```

### Ablation: RTN vs trained

`--epochs 0` skips training → plain round-to-nearest (RTN) quantization. Run it and
the full run with the *same* calib/eval (eval is deterministic) to isolate how much
Block-AP training buys you over RTN:

```bash
python run_gemma4_w4.py --epochs 0 ...   # RESULT [RTN (no train)] ...
python run_gemma4_w4.py --epochs 2 ...   # RESULT [trained 2ep] ...
```

### CPU validation (no GPU)

```bash
python run_gemma4_w4.py --device cpu --eval_device cpu \
  --train_size 16 --val_size 4 --epochs 1 --eval_seqs 6   # slow, correctness only
```

## Eval backends (`--eval_device`)

- `cuda`: **block-walk** — only pre-layers + one decoder block on GPU at a time
  (~7GB peak), so the 5.1B model evals on a 10GB card and ~10x faster than CPU.
- `cpu`: whole-model forward. Simple, but the full model won't fit a 10GB GPU, so
  this is the only CPU option (slow: minutes per window).

Both compute identical PPL: BOS-prefixed, deterministic non-overlapping windows,
loss over `seqlen` real tokens.

## Key facts (don't relearn these the hard way)

- **Load bf16, never fp16.** Gemma activations overflow fp16 → PPL ~1e4. `block_ap`
  follows the model's loaded dtype.
- **BOS is mandatory.** The tokenizer has `add_bos_token=False` and the model is
  extremely BOS-sensitive: no-BOS PPL is ~1000x inflated. Eval prepends it.
- **High absolute PPL (~60-95) is expected, not a bug.** `-it` is instruction-tuned
  (it injects chat/markdown formatting), so raw-corpus perplexity is intrinsically
  high. Generation is fluent and correct. Only the fp16→w4 **delta** matters for
  quantization quality — the baseline cancels.
- **C4 is streamed** (the new `datasets` lib can't load it by single-name id and a
  full-shard download is a RAM/disk hit). Deterministic first-`seqlen` crop per doc.

## Resources & scaling

Measured (RTX 3080, 10GB): peak ~6GB VRAM, block training ~3s/block. Per-layer-input
(PLE) is cached per batch — without that cache the projection reruns 35x per batch and
training takes hours.

CPU RAM budget ≈ model (~10GB) + activation cache (`2·train_size·seqlen·1536·2B`) +
PLE cache (~`train_size·seqlen·35·256·2B`) + shared_kv (in-memory). At
`train_size 256 / seqlen 1024` that's ~15GB. To go bigger: raise `train_size`/`seqlen`
for quality, and enable `off_load_to_disk` in `run_gemma4_w4.py`'s `qargs` once the
activation cache stops fitting RAM (shared_kv is not yet offloaded).

## Files

- `run_gemma4_w4.py` — entry point: load, eval, Block-AP, eval, report.
- `quantize/model_adapters.py` — `Gemma4Adapter` (PLE, KV-sharing, masks).
- `quantize/block_ap.py` — model-agnostic Block-AP loop (adapter-driven).
- `datautils_block.py` — calib/eval loaders (`get_c4` streams).
</content>
</invoke>
