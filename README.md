
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
*  `eumc.py`: This file contains the model of EUMC.
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

## Results

Our model achieves the following performance on(ASR | CA) :

|Dataset   | GTA | UGBA | DPGBA |EUMC|
| -------- |------- | --- | ---- |------ |
|     Cora     | 87.7\|77.4 | 83.1\|73.4 | 87.3\|82.5 | 97.4\|82.4 |
|    Pubmed    | 86.9\|84.8 | 88.9\|84.7 | 91.5\|85.3 | 96.4\|83.9 |
|     Flickr   | 93.1\|42.6 | 24.6\|43.4 | 53.5\|45.2 | 90.4\|44.5 |
|     Facebook | 76.0\|85.9 | 84.9\|85.8 | 86.6\|85.8 | 91.7\|83.8 |
|     Bicoin   | 79.2\|78.3 | 76.4\|78.3 | 80.1\|78.3 | 90.6\|78.3 |
|    OGB-arxiv | 68.4\|65.6 | 63.7\|64.9 | 68.8\|64.9 | 83.8\|65.3 |










