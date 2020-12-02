# Multi-level Distance Regularization for Deep Metric Learning

Official Code of AAAI 2021 paper "Multi-level Distance Regularization for Deep Metric Learning".

## Dependencies

You need a `CUDA-enabled GPU` and `python` (>3.6) to run the source code.

- torchvision >= 0.4.2
- torch >= 1.3.1
- tqdm
- scipy
- Pillow


```
pip install -r requirements.txt
```

## Preparing datasets
### 1. Make `dataset` directory 
```
mkdir ./dataset
```
### 2. (Optional) Only for In-Shop Clothes Retrieval
The source code will automatically download CUB-200-2011, Cars-196, and Stanford Online Products datasets.


But you need to manually download In-Shop Clothes Retrieval dataset.


1. Make `Inshop` directory in `./dataset` directory
```
mkdir -p ./dataset/Inshop
```
2. Download `img.zip` at the following link, and unzip it in `Inshop` directory
```
https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00
```
3. Download  `list_eval_partition.txt` at the following link, and put it in the `Inshop` directory.
```
https://drive.google.com/drive/folders/0B7EVK8r0v71pWVBJelFmMW5EWnM
```

##  Testing on the trained weights
```bash
# The models are trained with Triplet+MDR, please check Table 1.

# CUB-200-2011
wget https://github.com/anonymous-ai-research/pretrained/raw/master/cub200/cub200.pth
python run.py --mode eval --dataset cub200 --load cub200.pth

# Cars-196
wget https://github.com/anonymous-ai-research/pretrained/raw/master/cars196/cars196.pth
python run.py --mode eval --dataset cars196 --load cars196.pth

# Stanford Online Products
wget https://github.com/anonymous-ai-research/pretrained/raw/master/sop/sop.pth
python run.py --mode eval --dataset stanford --load sop.pth

# In-Shop Clothes Retrieval
wget https://github.com/anonymous-ai-research/pretrained/raw/master/inshop/inshop.pth
python run_inshop.py --mode eval --load inshop.pth
```

## Training
```bash
# CUB-200-2011
# Triplet
python run.py --dataset cub200 --lr 5e-5 --recall 1 2 4 8
# Triplet+L2Norm
python run.py --dataset cub200 --lr 5e-5 --recall 1 2 4 8 --l2norm
# Triplet+MDR
python run.py --dataset cub200 --lr 5e-5 --recall 1 2 4 8 --lambda-mdr 0.6 --nu-mdr 0.01
# Cars-196
# Triplet
python run.py --dataset cars196 --lr 5e-5 --recall 1 2 4 8
# Triplet+L2Norm
python run.py --dataset cars196 --lr 5e-5 --recall 1 2 4 8 --l2norm
# Triplet+MDR
python run.py --dataset cars196 --lr 5e-5 --recall 1 2 4 8 --lambda-mdr 0.2 --nu-mdr 0.01
# Stanford Online Products
# Triplet
python run.py --dataset stanford --num-image-per-class 3 --batch 256 --lr 1e-4 --recall 1 10 100 1000
# Triplet+L2Norm
python run.py --dataset stanford --num_image_per_class 3 --batch 256 --lr 1e-4 --recall 1 10 100 1000 --l2norm
# Triplet+MDR
python run.py --dataset stanford --num-image-per-class 3 --batch 256 --lr 1e-4 --recall 1 10 100 1000 --lambda-mdr 0.1 --nu-mdr 0.01
# In-Shop Clothes Retrieval
# Triplet
python run_inshop.py --num-image-per-class 3 --batch 256 --lr 1e-4 --recall 1 10 20 30 40 
# Triplet+L2Norm
python run_inshop.py --num-image-per-class 3 --batch 256 --lr 1e-4 --recall 1 10 20 30 40 --l2norm
# Triplet+MDR
python run_inshop.py --num-image-per-class 3 --batch 256 --lr 1e-4 --recall 1 10 20 30 40 --lambda-mdr 0.1 --nu-mdr 0.01
```
