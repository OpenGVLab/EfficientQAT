# llama-2-7b-w2g64
CUDA_VISIBLE_DEVICES=0 python trans_to_half_precision.py \
--resume_quant path/to/original/model \
--save_quant_dir path/to/new/half/precision/model \
--target_type fp16 \
--wbits 2 \
--group_size 64 \
--eval_ppl