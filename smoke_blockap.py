"""Smoke test: run block_ap end-to-end on gemma-4-E2B-it with tiny data. Confirms the
adapter-driven Block-AP runs through all blocks (incl. PLE + KV-sharing) without crashing
and keeps losses finite. ponytail: throwaway. Not a quality eval (that's a real PPL run).
"""
import types, torch, utils
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datautils_block import get_loaders
from quantize.block_ap import block_ap

MID = "google/gemma-4-E2B-it"
Path("./log").mkdir(exist_ok=True)
logger = utils.create_logger(Path("./log"))

args = types.SimpleNamespace(
    cache_dir="./cache", train_size=4, val_size=2, training_seqlen=544, batch_size=2,
    epochs=1, wbits=4, group_size=128, quant_lr=1e-4, weight_lr=1e-5, min_lr_factor=20,
    wd=0, clip_grad=0.3, early_stop=0, off_load_to_disk=False, real_quant=False, net="gemma4-smoke",
)
Path(args.cache_dir).mkdir(exist_ok=True)

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16, device_map="cpu")
for p in model.parameters(): p.requires_grad = False

# random calib data — smoke test checks the pipeline runs, not quality (avoids dataset-lib breakage)
V = model.config.get_text_config().vocab_size
def fake_loader(n):
    return [(torch.randint(0, V, (1, args.training_seqlen)), None) for _ in range(n)]
train, val = fake_loader(args.train_size), fake_loader(args.val_size)
block_ap(model, args, train, val, logger)
logger.info("SMOKE OK: block_ap completed all blocks")
print("SMOKE OK")
