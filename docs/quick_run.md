
# Get Started

## 1.Installation

Step1. Install Conda and create a Python 3.10 environment

```shell
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda create -n myenv python=3.10 --yes
conda activate myenv
```

Step1. Install YOLOX.

```shell
git clone https://github.com:dnoliver/YOLOX.git
cd YOLOX
pip install -r requirements.txt
pip install -e . --use-pep517 --no-build-isolation
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython;
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

## 2.Demo

Step1. Download a pretrained model from the benchmark table.

```shell
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

Step2. Use either -n or -f to specify your detector's config. For example:

```shell
python tools/demo.py image \
      --name yolox_s \
      --ckpt yolox_s.pth \
      --path ./assets/dog.jpg \
      --fp16 \
      --conf 0.25 \
      --nms 0.45 \
      --tsize 640 \
      --save_result \
      --device cpu
```
or

```shell
python tools/demo.py image \
      --exp_file ./exps/default/yolox_s.py \
      --ckpt yolox_s.pth \
      --path ./assets/dog.jpg \
      --fp16 \
      --conf 0.25 \
      --nms 0.45 \
      --tsize 640 \
      --save_result \
      --device cpu
```

Demo for video:

```shell
python tools/demo.py video \
    --name yolox_s \
    --ckpt yolox_s.pth \
    --path ./path/to/your/video \
    --conf 0.25 \
    --nms 0.45 \
    --tsize 640 \
    --save_result \
    --device cpu
```

## 3.Reproduce our results on COCO

Step1. Prepare COCO dataset
```shell
cd <YOLOX_HOME>
ln -s /path/to/your/COCO ./datasets/COCO
```

Step2. Reproduce our results on COCO by specifying -n:

```shell
python tools/train.py -n yolox-s -d 8 -b 64 --fp16 -o [--cache]
                         yolox-m
                         yolox-l
                         yolox-x
```
* -d: number of gpu devices
* -b: total batch size, the recommended number for -b is num-gpu * 8
* --fp16: mixed precision training
* --cache: caching imgs into RAM to accelarate training, which need large system RAM.

**Weights & Biases for Logging**

To use W&B for logging, install wandb in your environment and log in to your W&B account using

```shell
pip install wandb
wandb login
```

Log in to your W&B account

To start logging metrics to W&B during training add the flag `--logger` to the previous command and use the prefix "wandb-" to specify arguments for initializing the wandb run.

```shell
python tools/train.py -n yolox-s -d 8 -b 64 --fp16 -o [--cache] --logger wandb wandb-project <project name>
                         yolox-m
                         yolox-l
                         yolox-x
```

More WandbLogger arguments include

```shell
python tools/train.py .... --logger wandb wandb-project <project-name> \
                wandb-name <run-name> \
                wandb-id <run-id> \
                wandb-save_dir <save-dir> \
                wandb-num_eval_images <num-images> \
                wandb-log_checkpoints <bool>
```

More information available [here](https://docs.wandb.ai/guides/integrations/other/yolox).

**Multi Machine Training**

We also support multi-nodes training. Just add the following args:
* --num\_machines: num of your total training nodes
* --machine\_rank: specify the rank of each node

When using -f, the above commands are equivalent to:

```shell
python tools/train.py -f exps/default/yolox-s.py -d 8 -b 64 --fp16 -o [--cache]
                         exps/default/yolox-m.py
                         exps/default/yolox-l.py
                         exps/default/yolox-x.py
```

## 4.Evaluation

We support batch testing for fast evaluation:

```shell
python tools/eval.py -n  yolox-s -c yolox_s.pth -b 64 -d 8 --conf 0.001 [--fp16] [--fuse]
                         yolox-m
                         yolox-l
                         yolox-x
```
* --fuse: fuse conv and bn
* -d: number of GPUs used for evaluation. DEFAULT: All GPUs available will be used.
* -b: total batch size across on all GPUs

To reproduce speed test, we use the following command:
```shell
python tools/eval.py -n  yolox-s -c yolox_s.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse
                         yolox-m
                         yolox-l
                         yolox-x
```
