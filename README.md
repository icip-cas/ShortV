# ShortV
Code release for "[ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers](https://arxiv.org/abs/2504.00502)"

## Experiments Environment
### Set Up the Dependencies as:
```bash
# install llava
conda create -n shortv python=3.10 -y
conda activate shortv
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
# install lmms-eval
cd lmms-eval
pip install -e .
```

## Chatbot
```bash
export SKIP_LAYERS="31,29,30,28,0,26,27,25,24,22,23,21,2,3,20,18,17,12,19"
python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b   \
    --image-file "https://llava-vl.github.io/static/images/view.jpg"
``` 

## Evaluate ShortV
```bash
export SKIP_LAYERS="31,29,30,28,0,26,27,25,24,22,23,21,2,3,20,18,17,12,19"
accelerate launch  --num_processes=1 --main_process_port=12346 -m lmms_eval --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b"  \
    --tasks mmmu_val --batch_size 1 \
    --log_samples_suffix llava_7b \
    --output_path ./logs/7b/ 
```

## Acknowledge
This work is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and [VTW](https://github.com/lzhxmu/VTW)