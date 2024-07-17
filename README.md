# EfficientQAT
Official PyTorch implement of paper [EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](https://arxiv.org/abs/2407.11062)

## News
- Note: The end-to-end inference speedup through [BitBLAS](https://github.com/microsoft/BitBLAS) is working in progress. Current code can obtain actually reduction of training and inference memory footprint, but can not speedup inference.
- [2024/07] We release EfficientQAT, which pushes the limitation of uniform (INT) quantization in an efficient manner.

## Contents
- [Install](#install)
- [Model Zoo](#model-zoo)
- [Training](#training)
- [Inference](#Inference)
- [Citation](#citation)


## Install
1. Clone this repository and navigate to EfficientQAT folder
```
git clone https://github.com/OpenGVLab/EfficientQAT.git
cd EfficientQAT
```

2. Install package
```
conda create -n efficientqat python==3.9

conda activate efficientqat

pip install -r requirements.txt
```

## Model Zoo

We provide a number of prequantized EfficientQAT models as follows: 

- WikiText2 PPL is measured in 2048 context length.
- Avg. Accuracy indicate the average accuracy in 5 zero-shot reasoning tasks (WinoGrande,PIQA,HellaSwag,Arc-Easy, Arc-Challenge) with [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) v0.4.2.
- 1GB = $10^9$ Bit

| Model | Quantization | WikiText2 PPL | Avg. Accuracy | Model Size (GB) | Hub link|
|-------|--------------|---------------|---------------|-----------------|----------|
Llama-2-7B|fp16|5.47|64.86|13.2|-|
Llama-2-7B|w4g128|5.53|64.27|3.7|[Link](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w4g128)|
Llama-2-7B|w3g128|5.81|64.02|3.1|[Link](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w3g128)|
Llama-2-7B|w2g64|6.86|60.14|2.3|[Link](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w2g64)|
Llama-2-7B|w2g128|7.17|59.50|2.2|[Link](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w2g128)|
Llama-2-13B|fp16|4.88|67.81|25.4|-|
Llama-2-13B|w4g128|4.93|67.52|6.8|[Link](https://huggingface.co/ChenMnZ/Llama-2-13b-EfficientQAT-w4g128)|
Llama-2-13B|w3g128|5.12|67.28|5.6|[Link](https://huggingface.co/ChenMnZ/Llama-2-13b-EfficientQAT-w3g128)|
Llama-2-13B|w2g64|5.96|64.88|4.0|[Link](https://huggingface.co/ChenMnZ/Llama-2-13b-EfficientQAT-w2g64)|
Llama-2-13B|w2g128|6.08|63.88|3.8|[Link](https://huggingface.co/ChenMnZ/Llama-2-13b-EfficientQAT-w2g128)|
Llama-2-70B|fp16|3.32|72.41|131.6|-|
Llama-2-70B|w4g128|3.39|72.62|35.8|[Link](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w4g128)|
Llama-2-70B|w3g128|3.61|71.76|29.1|[Link](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w3g128)|
Llama-2-70B|w2g64|4.52|69.48|20.1|[Link](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w2g64)|
Llama-2-70B|w2g128|4.61|68.93|18.9|[Link](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w2g128)|
Llama-3-8B|fp16|6.14|68.58|13.0|-|
Llama-3-8B|w4g128|6.47|68.43|5.4|[Link](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w4g128)|
Llama-3-8B|w3g128|7.09|67.35|4.7|[Link](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w3g128)|
Llama-3-8B|w2g64|9.41|60.76|3.9|[Link](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w2g64)|
Llama-3-8B|w2g128|9.80|59.36|3.8|[Link](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w2g128)|
Llama-3-70B|fp16|2.85|75.33|137.8|-|
Llama-3-70B|w4g128|3.17|74.57|38.9|[Link](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w4g128)|
Llama-3-70B|w3g128|4.19|72.42|32.2|[Link](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w3g128)|
Llama-3-70B|w2g64|6.08|67.89|23.2|[Link](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w2g64)|
Llama-3-70B|w2g128|6.38|67.57|22.0|[Link](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w2g128)|
Llama-3-8B-Instruct|fp16|8.29|68.43|13.0|-|
Llama-3-8B-Instruct|w4g128|7.37|68.69|5.4|[Link](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w4g128)|
Llama-3-8B-Instruct|w3g128|7.92|66.75|4.7|[Link](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w3g128)|
Llama-3-8B-Instruct|w2g64|10.22|60.79|3.9|[Link](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g64)|
Llama-3-8B-Instruct|w2g128|10.73|59.74|3.8|[Link](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g128)|
Llama-3-70B-Instruct|fp16|5.33|73.78|137.8|-|
Llama-3-70B-Instruct|w4g128|3.90|74.33|38.9|[Link](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w4g128)|
Llama-3-70B-Instruct|w3g128|4.66|73.71|32.2|[Link](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w3g128)|
Llama-3-70B-Instruct|w2g64|6.60|69.75|23.2|[Link](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w2g64)|
Llama-3-70B-Instruct|w2g128|6.89|67.79|22.0|[Link](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w2g128)|

## Training
EfficientQAT involves two consecutive training phases: Block-wise training of all parameters (**Block-AP**) and end-to-end training of quantization parameters (**E2E-QP**). The detailed training script can be found in `./examples`. We give the training script examples on Llama-2-7B with w2g64 quantization in the following. 

1. Block-AP

You should modify `--model` to the folder of full-precision model  in the script before you running the following command.
```
bash examples/block_ap/Llama-2-7b/w2g64.sh
```
Specifically, the `--weight_lr` is `2e-5` for 2-bit and `1e-5` for 3-/4-bits in our experiments.

Some other important arguments:
- `--train_size`: number of training data samples, 4096 as default
- `--val_size`: number of validation data samples, 64 as default
- `--off_load_to_disk`: save training dataset to disk, saving CPU memory but may reduce training speed


2. E2E-QP

Then, you can load the quantized model of Block-AP for further E2E-QP. Specifically, E2E-QP can adapt to different scenarios by changing the training datasets. You should modify `--quant_model_path` to the folder of quantized model in the script before you running the following command.

1) Train on RedPajama
```
bash examples/e2e_qp/Llama-2-7b/w2g64-redpajama.sh
``` 

2) Train on Alpaca
```
bash examples/e2e_qp/Llama-2-7b/w2g128-redpajama.sh
```
Specifically, the `--learning_rate` is `2e-5` for 2-bit and `1e-5` for 3-/4-bits in our experiments. You can decrease the `--per_device_train_batch_size` to reduce the memory footprint during trainin, and making sure that `--gradient_accumulation_steps`  increases by the same multiple to maintain the same batch size.

After E2E-QP, you can also leverage `trans_to_half_precision.py` to further reducing the model size through transfer some float32 data into half-precision ones:  
```
bash examples/model_transfer/llama-2-7b.sh
```

## Inference

1. Download the pre-quantized EfficientQAT models from Huggingface
```
pip install huggingface_hub

huggingface-cli download ChenMnZ/Llama-2-7b-EfficientQAT-w2g64 --local-dir ./output/pre_quantized_models/Llama-2-7b-EfficientQAT-w2g64
```

2. Evaluate the pre-quantized EfficientQAT model
```
CUDA_VISIBLE_DEVICES=0 python main_block_ap.py \
--resume_quant ./output/pre_quantized_models/Llama-2-7b-EfficientQAT-w2g64 \
--net Llama-2 \
--wbits 2 \
--group_size 64 \
--output_dir ./output/inference_results/ \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande
```

## Citation
If you found this work useful, please consider citing:
```
@article{efficientqat,
  title={EfficientQAT: Efficient Quantization-Aware Training for Large Language Models},
  author={Chen, Mengzhao and Shao, Wenqi and Xu, Peng and Wang, Jiahao and Gao, Peng and Zhang, Kaipeng and Qiao, Yu and Luo, Ping},
  journal={arXiv preprint arXiv:2407.11062},
  year={2023}
}
```