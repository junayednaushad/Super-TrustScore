# Super-TrustScore

## Dependencies
- python == 3.10.12
- torch == 2.0.1
- pytorch-lightning == 2.1.0
- CUDA == 11.3
- wandb == 0.15.12
```
# clone this repository
git clone git@github.com:junayednaushad/Super-TrustScore.git
cd Super-TrustScore

# create conda environment and install dependencies
conda create --name STS
conda activate STS
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install lightning -c conda-forge
pip install transformers
conda install pandas
conda install -c conda-forge matplotlib
conda install -c anaconda scikit-learn
conda install -c conda-forge wandb
conda install -c anaconda seaborn

# make directories
mkdir configs data figures inference_results models
```

## Skin Lesion Datasets
### HAM10k (In-Distribution)
Use the following links to download the data from the official challenge website to ./data/HAM10k/:
- [Training Images](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip)
- [Training Lesion Groupings](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_LesionGroupings.csv)
- [Training Ground Truth](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip)
- [Test Images](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_Input.zip)
- [Test Ground Truth](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_GroundTruth.zip)

We chose not to use the validation data because there are no lesion groupings provided. This means that there is no way to confirm that the images in the validation set are not of the same lesion as those found in the training data. Also, the validation set is very small (i.e., only 1 instance of Dermatofibroma and 3 instances of Vascular Lesion) so it can't be used as a stand-alone dataset to accurately evaluate the learning ability of the model during training. Instead, we perform a stratified split of the training data while making sure that all images in the validation set are from lesions not contained in the training set.

```
# unzip the downloaded data and create the folder /data/HAM10k before running the preprocessing code
python src/preprocessing/ham10k.py
```

### ISIC 2019 (Shifted-Distribution I)
Download the [Task 1 Training Data](https://challenge.isic-archive.com/data/#2019)
```
# make sure you've already downloaded the HAM10k data and preprocessed it before preprocessing ISIC 2019 (we remove any images in ISIC 2019 that overlap with HAM10k)
# unzip and move data to /data/ISIC_2019
python src/preprocessing/isic_2019.py
```

### PAD-UFES-20 (Shifted-Distribution II)
Download images and metadata from [here](https://data.mendeley.com/datasets/zr7vgbcyr2/1)
```
python src/preprocessing/pad_ufes_20.py
```

## Diabetic Retinopathy Datasets
### EyePACS (In-Distribution)
Download the data from the [Kaggle Diabetic Retinopathy Detection](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data) competition

```
# unzip train and test data and put them in /data/EyePACS/train & /data/EyePACS/test before preprocessing
python src/preprocessing/eyepacs.py
```

### Messidor-2 (Shifted-Distribution I)
Download the data from [here](https://www.adcis.net/en/third-party/messidor2/)

```
python src/preprocessing/messidor_2.py
```

### APTOS 2019 (Shifted-Distribution II)
Download the data from [here](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)

```
python src/preprocessing/aptos_2019.py
```

## Training
In order to train a Swin-S model on the HAM10k dataset you need to create a config file at this location: `configs/train/HAM10k/swin_s.yaml`. Example config file:
```
project: Super-TrustScore
dataset: HAM10k
run_name: swin_s
seed: 100
num_classes: 7
return_filename: False
data_dir: ../data/HAM10k/
model_dir: ../models/HAM10k/swin_s/

train_transforms: |
  T.Compose([
              T.RandomResizedCrop(size=(224,298), scale=(0.5,1), ratio=(1, 1.5)),
              T.TrivialAugmentWide(),
              T.RandomHorizontalFlip(0.33),
              T.RandomVerticalFlip(0.2),
              T.ToTensor(),
              T.Normalize(
                (0.6523304, 0.62197226, 0.61544853),
                (0.11362693, 0.16195466, 0.16857147)
              )
  ])
test_transforms: |
  T.Compose([
              T.Resize((224,298)),
              T.ToTensor(),
              T.Normalize(
                (0.6523304, 0.62197226, 0.61544853),
                (0.11362693, 0.16195466, 0.16857147)
              )
  ])

pretrained: True
batch_size: 32
num_workers: 1
pin_memory: True
epochs: 30
lr: 0.0001
weight_decay: 0.05
momentum: null
```

Then train the model:
```
cd src
python train.py --config ../configs/train/HAM10k/swin_s.yaml
```

## Inference
Inference involves saving model predictions, softmax probabilities, and embeddings. Running inference also requires a config file:

```
dataset: HAM10k
num_classes: 7
only_test: False # set to True for shifted-distribution datasets

ckpt_path: ../models/HAM10k/swin_s/model.ckpt
save_path: ../inference_results/HAM10k/swin_s.npy

MCDropout: True # setting this to True if you want inference results for MCD
num_inferences: 10 # number of forward passes

data_dir: ../data/HAM10k/
return_filename: True
train_transforms: |
  T.Compose([
              T.Resize((224,298)),
              T.ToTensor(),
              T.Normalize(
                (0.6523304, 0.62197226, 0.61544853),
                (0.11362693, 0.16195466, 0.16857147)
              )
  ])
test_transforms: |
  T.Compose([
              T.Resize((224,298)),
              T.ToTensor(),
              T.Normalize(
                (0.6523304, 0.62197226, 0.61544853),
                (0.11362693, 0.16195466, 0.16857147)
              )
  ])
batch_size: 32
num_workers: 1
pin_memory: True
```

Run inference:

```
python inference.py --config ../configs/inference/HAM10k/swin_s.yaml
```

## Benchmark Confidence Scoring Functions
Calculate AURC and Risk@50 for different confidence scoring functions. Example config file:

```
dataset: HAM10k
SD: False
iid_inference_results_dir: ../inference_results/HAM10k # directory containing inference results
iid_inference_files: # list of files containing inference results (if multiple files provided then metrics will be averaged over all files)
  - swin_s.npy

confidence_scoring_functions: # List of CSFs to benchmark
  - Softmax
  - MCDropout
  - DeepEnsemble
  - ConfidNet
  - TrustScore
  - Mahalanobis
  - Local
  - Global
  - Super-TrustScore

get_scores: # Each boolean corresponds to the CSF in the previous list (same order) 
  - True # If True calculate confidence scores, if False then inference file should already contain confidence scores to be loaded
  - True
  - True
  - True
  - True
  - True
  - True
  - True
  - True

# Classification Performance
get_classification_performance: True
clf_metrics:
  - Balanced Accuracy
  - Accuracy


# Embedding quality
get_clustering_metrics: True
clustering_reduce_dim: True
clustering_n_components: 0.9
clustering_norm: True

# TrustScore hyperparameters
ts_reduce_dim: True
ts_n_components: 0.9
ts_norm: True
ts_filtering: none
ts_num_workers: 4

# Mahalanobis hyperparameters
mahal_norm: True
mahal_reduce_dim: True
mahal_n_components: 0.9

# Super-TrustScore hyperparameters
knn_reduce_dim: True
knn_n_components: 0.9
knn_norm: True
knn_filtering: False
local_conf: True
global_conf: True
min_k: 1
max_k: 20
k_step: 1
N_samples: 1014
eps: 0

# plot risk coverage plot
plot_rc: True
coverage: 0.5
plot_title: HAM10k (ID)
plot_dir: ../figures/HAM10k/
```

Run benchmark:
```
python benchmark.py --config ../configs/benchmark/HAM10k/swin_s.yaml
```
