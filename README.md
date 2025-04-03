# ShortV
Code release for "[ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers](https://arxiv.org/abs/2504.00502)"

## Install

1. Clone this repository and navigate to ShortV folder
```bash
git clone https://github.com/icip-cas/ShortV.git
cd ShortV
```

2. Install Package
```bash
conda create -n shortv python=3.10 -y
conda activate shortv
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for evaluation with lmms-eval
```bash
cd lmms-eval
pip install -e .
```

## ShortV Inference and Evaluation

### Replaced Layers

The layer ids of replaced layers are provided below.

| Model | Checkpoint | Replaced Layers |
| --- | --- | --- |
| LLaVA-1.5-7B | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) | 31,29,30,28,0,26,27,25,24,22,23,21,2,3,20,18,17,12,19 |
| LLaVA-1.5-13B | [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) | 39,32,28,36,27,37,29,30,1,38,25,31,2,26,23,34,0,33,35,22,24,21,20,17 |
| LLaVA-NeXT-7B | [liuhaotian/llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) | 31,29,30,28,26,27,22,24,21,23,25,20,19,17,18,15,12,0,2 |
| LLaVA-NeXT-13B | [liuhaotian/llava-v1.6-vicuna-13b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b) | 39,32,29,36,27,30,37,23,25,31,26,2,28,22,33,35,34,24,38,21,20,18,1,17 |


### Chatbot Inference

Chat about images using ShortV. 

```bash
export REPLACED_LAYERS="31,29,30,28,0,26,27,25,24,22,23,21,2,3,20,18,17,12,19"
python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b   \
    --image-file "https://llava-vl.github.io/static/images/view.jpg"
```

### Evaluation with LMMs-Eval

[LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) is an evaluation framework meticulously crafted for consistent and efficient evaluation of LMM.

```bash
export MODEL_PATH="liuhaotian/llava-v1.5-7b"
export MODEL_NAME="llava_7b"
export CONV_MODE="v1"
export REPLACED_LAYERS="31,29,30,28,0,26,27,25,24,22,23,21,2,3,20,18,17,12,19"
accelerate launch  --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model llava \
    --model_args pretrained=${MODEL_PATH},conv_template=${CONV_MODE}  \
    --tasks mmmu_val \
    --batch_size 1 \
    --log_samples_suffix ${MODEL_NAME} \
    --output_path ./logs/ 
```

### Evaluation with Scripts From LLaVA

See [Evaluation.md](https://github.com/icip-cas/ShortV/blob/main/docs/Evaluation.md).

## Calculating LC Scores and Identifying Ineffective Layers

To identify which layers are ineffective, we calculate visual LC scores for all MLLM layers.

```bash
export MODEL_PATH="liuhaotian/llava-v1.5-7b"
export MODEL_NAME="llava_7b"
export CONV_MODE="v1"
accelerate launch  --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model llava \
    --model_args pretrained=${MODEL_PATH},conv_template=${CONV_MODE}  \
    --tasks gqa,flickr30k_test \
    --batch_size 1 \
    --log_samples_suffix ${MODEL_NAME} \
    --output_path ./logs/ \
    --limit 20 \
    --cal_lc
```

You will get visual LC scores for each layer, and the order of layer replacement.

## Acknowledge
This work is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and [VTW](https://github.com/lzhxmu/VTW)

## Citation
If you find ShortV useful for your research and applications, please cite using this BibTeX:
```bib
@misc{yuan2025shortvefficientmultimodallarge,
      title={ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers}, 
      author={Qianhao Yuan and Qingyu Zhang and Yanjiang Liu and Jiawei Chen and Yaojie Lu and Hongyu Lin and Jia Zheng and Xianpei Han and Le Sun},
      year={2025},
      eprint={2504.00502},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.00502}, 
}
```