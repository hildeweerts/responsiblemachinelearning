(mlpreliminaries)=
# Machine Learning Preliminaries 

This chapter provides a brief introduction to various machine learning concepts that are required to understand the technical contents in this book.

```{admonition} Summary
:class: tip

**Machine Learning**

{term}`Machine Learning` is a subfield of the broader field of artificial intelligence. It has three main paradigms: {term}`supervised learning`, {term}`unsupervised learning`, and {term}`reinforcement learning`. 

The main focus of this book is on {term}`supervised learning`, in which the learning tasks consists of predicting a {term}`target variable` from a set of input {term}`feature`s. We focus on traditional machine learning algorithms (as opposed to deep learning).

**Model Evaluation**

We define predictive performance metrics based on the {term}`confusion matrix`, which visualizes different types of classification errors. We explain the metrics: {term}`accuracy`, {term}`false positive rate`, {term}`true negative rate`, {term}`false negative rate`, {term}`true positive rate`, {term}`recall`, {term}`precision`, and {term}`F1-score`.

We discuss ways to evaluate predictive performance based on the model's conficence score. We introduce the {term}`ROC curve`, which sets out the {term}`true positive rate` against the {term}`false positive rate` and can be summarized using {term}`AUC` score. Additionally, we discuss the {term}`precision-recall curve`, which sets out a model's {term}`precision` against {term}`recall` and can be summarized by computing the {term}`average precision`.

Model {term}`calibration` refers to the extent to which the confidence scores correspond to actual probabilities.

**Model Selection**

We discuss several best practices in model selection: the process of choosing among different candidate models. We discuss how a model's predictive performance can be estimated more accurately by splitting the data into a {term}`training set`, {term}`validation set` and {term}`test set`. Moreover, we discussed how this procedure can be further extended to a {term}`k-fold cross-validation`, {term}`repeated cross-validation`, and {term}`nested cross-validation`. Additionally, we briefly discussed two common approaches for {term}`hyperparameter` tuning: `grid search` (an exhaustive search over the search space) and {term}`random search` (random sampling of hyperparameter settings from the search space). 

**Cost-sensitive Learning**

We explore several approaches to {term}`cost-sensitive learning`, in which different types of mistakes may be associated with different costs. In particular, we show how a classifier's {term}`decision threshold` can be optimized to account for different costs.
```

(intromlfurtherreading)=
```{admonition} Further Reading
:class: seealso

* *An Introduction to Machine Learning with Python* by Andreas C. Müller and Sarah Guido. A very practical introduction to machine learning using the popular Python library scikit-learn.

* *Pattern Recognition and Machine Learning* by Christopher Bishop. If you are looking for a deeper understanding of machine learning algorithms, including thorough mathematical details, this book is the way to go. I have heard some people refer to it as the Machine Learning Bible.
```