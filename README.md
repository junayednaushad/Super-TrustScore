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

## Inference
Save model predictions, softmax probabilities, and embeddings.

## Benchmark Confidence Scoring Functions
