
# Attack by Yourself: Effective and Unnoticeable Multi-Category Graph Backdoor Attacks with Subgraph Triggers Pool


## Introduction

EUMC is an innovative framework designed to address the complex challenge of conducting effective and unnoticeable multi-category graph backdoor attacks on node classification.
 

## Project Structure

*  `utils.py`: The functions to select nodes, load and split data.
*  `eumc.py`: This file contains the model of EUMC.
*  `construct.py`: This file contains target model info.
*  `surrogate_models`: The framework of baseline backdoor attack.
*  `train.py`: The program to run our EUMC attack.

## Get Started
First, download our repo
```
git clone https://github.com/justincui03/or-bench
cd or-bench
```
Next, install the required environment
```setup
conda create -n EUMC python=3.9
conda activate EUMC

# Install scikit-learn-extra
pip install scikit-learn-extra==0.3.0

# Install Pytorch with CUDA support
# Please adjust the CUDA version if necessary
conda install pytorch==2.21 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install torch_geometric
pip install torch_geometric==2.5.0

# Install ogb
pip install ogb==1.3.6

# Install additional dependencies
pip install pandas torchmetrics Deprecated
```
## Data
The experiments are conducted on six public real-world datasets, i.e., Cora, Pubmed, Facebook, Bitcoin, Flickr, and OGB-arxiv which can be automatically downloaded to `./data` through torch-geometric API.

## Training

To train EUMC , run this command:

```train
python train.py
```

You can set the dataset parameter in  `train.py` to achieve the evaluation of specific datasets. The specific parameters are elaborated in detail in the paper.

## Evaluation

To evaluate our model on target dataset, run:

```eval
python train.py --dataset Cora --eval
```
You can set the dataset and eval parameters in `train.py` to achieve the evaluation of specific datasets.











