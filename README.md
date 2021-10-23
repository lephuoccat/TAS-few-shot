# Task Affinity with Maximum Bipartite Matching in Few-Shot Learning
This is the source code for Task Affinity with Maximum Bipartite Matching in Few-Shot Learning paper (https://arxiv.org/pdf/2110.02399.pdf).

## Description

We propose an asymmetric affinity score for representing the complexity of utilizing the knowledge of one task for learning another one. Our method is based on the maximum bipartite matching algorithm and utilizes the Fisher Information matrix. We provide theoretical analyses demonstrating that the proposed score is mathematically well-defined, and subsequently use the affinity score to propose a novel algorithm for the few-shot learning problem. In particular, using this score, we find relevant training data labels to the test data and leverage the discovered relevant data for episodically fine-tuning a few-shot model. Results on various few-shot benchmark datasets demonstrate the efficacy of the proposed approach by improving the classification accuracy over the state-of-the-art methods even when using smaller models.


<p align="center">
  <img src="images/fig1.jpg" height="550" title="procedure">
</p>

## Getting Started

### Dependencies

* Requires Pytorch, Numpy
* miniImageNet (https://github.com/yaoyao-liu/mini-imagenet-tools)
* tieredImageNet (https://github.com/yaoyao-liu/tiered-imagenet-tools)
* CIFAR-FS (https://github.com/bertinetto/r2d2)
* FC-100 (https://github.com/ElementAI/TADAM)

### Executing program

* First, we train the whole classifier with the entire training set.
```
python train_classifier.py
```
* Next, we define base tasks from the training set, and extract the mean hidden features for each of the class of data in both training and test set. Note that, the test set only has a few-shot data.
```
python center_feature.py
```
* Then, we compute the task affinity using the Fisher task distance with the maximum bipartite matching algorithm. 
```
python matching_Fisher_distance.py
```
* After obtaining the related base tasks (using the computed task affinity), we use the base tasks' data samples to fine-tune the classifier. The few-shot tasks from the test set are also used to fine-tune the classifier.
```
python train_meta.py
```
* Lastly, the classifier is evaluated with a series of test tasks generated from the test set.
```
python test_few_shot.py
```

### Results
The distribution of TAS found in miniImageNet (left) and the frequency of 64 classes in thetop-8 closest source tasks (right) in miniImageNet.
<p align="center">
  <img src="images/fig2.jpg" height="300" title="dis1">
</p>

The distribution of TAS found in tieredImageNet (left) and the frequency of 351 classes in the top-6 closest source tasks (right) in tieredImageNet.
<p align="center">
  <img src="images/fig3.jpg" height="300" title="dis2">
</p>

The table below indicates the performance of our method for 5-way 1-shot and 5-way 5-shot classification with 95% confidence interval on miniImageNet dataset
| Model        | Backbone      | Paramameters (M) | 1-shot      |  5-shot   |
| :---         |    :---:      |     :---:        |  :---:      | :---:     |
| TAS-simple   | ResNet-12     |  7.99            | 64.71±0.43  | 82.08±0.45|
| TAS-distill  | ResNet-12     |  7.99            | 65.13±0.39  | 82.47±0.52|
| TAS-distill+ | ResNet-12     |  12.47           | 65.68±0.45  | 83.92±0.55|


## Authors

Cat P. Le (cat.le@duke.edu), 
<br>Juncheng Dong, 
<br>Mohammadreza Soltani, 
<br>Vahid Tarokh