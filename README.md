# Attack by Yourself: Effective and Unnoticeable Multi-Category Graph Backdoor Attacks with Subgraph Triggers Pool


## Introduction

EUMC is an innovative framework designed to address the complex challenge of conducting effective and unnoticeable multi-category graph backdoor attacks on node classification.


## Project Structure

*  `utils.py`: The functions to select nodes, load and split data.
*  `eumc.py`: This file contains the model of EUMC.
*  `construct.py`: This file contains target model info.
*  `surrogate_models`: The framework of baseline backdoor attack.
*  `clustering_nodes`: The algorithm of clustering nodes.
*  `train.py`: The program to run our EUMC attack.

## Get Started
First, download our repo
```
git clone https://github.com/novdream/EUMC
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

## Training and Test

To train EUMC , run this command:

```train
bash train.sh
```

You can set the dataset parameter in  `train.sh` to achieve the evaluation of specific datasets. The specific parameters are elaborated in detail in the paper.


## Reference
If you find our code useful for your research, please consider citing our paper.
```
@article{cui2024or,
  title={OR-Bench: An Over-Refusal Benchmark for Large Language Models},
  author={Cui, Justin and Chiang, Wei-Lin and Stoica, Ion and Hsieh, Cho-Jui},
  journal={arXiv preprint arXiv:2405.20947},
  year={2024}
}
```