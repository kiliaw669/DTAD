# Anomaly Detection with Conditioned Denoising Diffusion Models.

Official implementation of DTAD

## Requirements
This repository is implemented and tested on Python 3.10 and PyTorch 2.5.1
To install requirements:

```setup
conda create -n dtad python=3.10
conda activate dtad
pip install -r requirements.txt

# xformers is optional, but it would greatly speed up the attention computation.
pip install -U xformers
pip install -U --pre triton
```

## Train and Evaluation of the Model
You can download the model checkpoints directly from [Checkpoints](https://drive.google.com/drive/u/0/folders/1FF83llo3a-mN5pJN8-_mw0hL5eZqe9fC) 

To train the denoising UNet, run:

```train
python main.py --train True
```

Modify the settings in the config.yaml file to train the model on different categories.

For fine-tuning the feature extractor, use the following command:

```domain_adaptation
python main.py --domain_adaptation True
```

To evaluate and test the model, run:

```detection
python main.py --detection True
```

## Dataset
You can download  [MVTec AD: MVTec Software](https://www.mvtec.com/company/research/datasets/mvtec-ad/) and [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) Benchmarks.
For preprocessing of VisA dataset check out the [Data preparation](https://github.com/amazon-science/spot-diff/tree/main) section of this repository.

The dataset should be placed in the 'datasets' folder. The training dataset should only contain one subcategory consisting of nominal samples, which should be named 'good'. The test dataset should include one category named 'good' for nominal samples, and any other subcategories of anomalous samples. It should be made as follows:

```shell
Name_of_Dataset
|-- Category
|-----|----- ground_truth
|-----|----- test
|-----|--------|------ good
|-----|--------|------ ...
|-----|--------|------ ...
|-----|----- train
|-----|--------|------ good
```

## Acknowledgements
We thank the great works [DDAD](https://arxiv.org/abs/2305.15956) and [U-ViT](https://arxiv.org/abs/2209.12152) for providing assistant codes for our research.