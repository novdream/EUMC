
# Attack by Yourself: Effective and Unnoticeable Multi-Category Graph Backdoor Attacks with Subgraph Triggers Pool


## Introduction

EUMC is an innovative framework designed to address the critical issue of crafting effective and unnoticeable multi-category graph backdoor attacks. 
 

**Components**
- **Multi-Category Subgraph Triggers Pool:** Constructs a set of diverse triggers that maintain the intrinsic structural properties of the data and are aware of different categories.
- **Trigger Attachment Strategy:** Develops a "select then attach" strategy that connects suitable category-aware triggers to attacked nodes for unnoticeability.


**Contributions:**
-  **Problem Exploration:** We focus on the challenging problem of effective and unnoticeable multi-category graph backdoor attack
-  **Method Development:** We design a novel framework that constructs a multi-category subgraph trigger pattern (MC-STP) from the attacked graph. We also develop a “select then attach” strategy to ensure the unnoticeability and effectiveness of the multi-category graph backdoor attacks.

## Overview

*  `utils.py`: The functions to select nodes, load and split data.
*  `emuc.py`: This file contains the model of EUMC.
*  `construct.py`: This file contains target model info.
*  `surrogate_models`: The framework of baseline backdoor attack.
*  `train.py`: The program to run our EUMC attack.


## Requirements

To install requirements:

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
The experiments are conducted on six public real-world datasets, i.e., Cora, Pubmed, Facebook, Bitcoin, Flickr, OGB-arxiv which can be automatically downloaded to `./data` through torch-geometric API.

## Training

To train EUMC , run this command:

```train
python train.py
```

You can set the dataset parameter in  train.py to achieve the evaluation of specific datasets. The specific parameters are elaborated in detail in the paper.

## Evaluation

To evaluate our model on target dataset, run:

```eval
python train.py --dataset Cora --eval
```
You can set the dataset and eval parameters in train.py to achieve the evaluation of specific datasets.

## Results

Our model achieves the following performance on(ASR | CA) :

|Dataset   | GTA | UGBA | DPGBA |EUMC|
| -------- |------- | --- | ---- |------ |
|     Cora     |  0.75 \| 0.82 | 0.76 \| 0.82 | 0.78 \| 0.81 | 0.97 \| 0.81 |
|    Pubmed    | 0.25 \| 0.69 | 0.51 \| 0.70 | 0.09 \| 0.71 | 0.91 \| 0.76 |
|     Flickr   |    GPL | 0.51 \| 0.21 | 0.63 \| 0.26 | 0.46 \| 0.29 | 0.99 \| 0.34 |
|     Facebook |    GSL | 0.79 \| 0.87 | 0.79 \| 0.86 | 0.66 \| 0.87 | 0.96 \| 0.84 |
|  Bicoin      |    GCL | 1.00 \| 0.20 | 0.67 \| 0.84 | 0.23 \| 0.84 | 0.93 \| 0.84 |
|    OGB-arxiv |    GPL | 0.54 \| 0.39 | 0.65 \| 0.50 | 0.82 \| 0.45 | 1.00 \| 0.44 |










