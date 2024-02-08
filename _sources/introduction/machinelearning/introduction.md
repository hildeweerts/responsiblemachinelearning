(ml_preliminaries)=

# Machine Learning Preliminaries

This chapter provides a brief introduction to various machine learning concepts that are required to understand the technical contents in this book. The focus of this book is on supervised learning, an area of machine learning aimed towards classification or prediction. At an abstract level, a supervised machine learning model can be seen as a function that maps a set of input features to the corresponding target variable output. For example, we might want to train a model that is able to predict whether an e-mail (consisting of several features such as whether the e-mail contains certain words) is spam (target variable). The task of the machine learning algorithm is to find the best mapping between the input features and the target variable.

Supervised learning can be further categorized based on the target variable type. In regression problems, the target variable is continuous. In classification problems, the target variable is a categorical variable. The different categories are referred to as classes. The binary classification problem is a particular type of classification problems in which the target variable can take two values. The two classes are usually represented as 0 and 1 and referred to as the negative class and positive class respectively.

```{note}
The distinction between the positive and negative class merely indicates what we are trying to detect and does not imply sentiment. For example, in the case of fraud detection, we are interested in detecting fraudulent transactions. A fraudulent transaction is therefore often defined to belong to the positive class - even though fraud is typically not deemed desirable.
```

```{admonition} Summary
:class: tip

**Machine Learning**

Machine Learning is a subfield of the broader field of artificial intelligence. The main focus of this book is on supervised learning, in which the learning tasks consists of predicting a target variable from a set of input features. We focus on traditional machine learning algorithms (as opposed to deep learning).

[**Model Evaluation**](model_evaluation)

We define predictive performance metrics based on the confusion matrix, which visualizes different types of classification errors. We explain several predictive performance metrics: accuracy, false positive rate, true negative rate, false negative rate, true positive rate, recall, precision, and F1-score.

We discuss ways to evaluate predictive performance based on the model's conficence score. We introduce the ROC curve, which sets out the true positive rate against the false positive rate and can be summarized using AUC score. Additionally, we discuss the precision-recall curve, which sets out a model's precision against recall and can be summarized by computing the average precision.

Model calibration refers to the extent to which the confidence scores correspond to actual probabilities.

[**Model Selection**](model_selection)

We discuss several best practices in model selection: the process of choosing among different candidate models. We discuss how a model's predictive performance can be estimated more accurately by splitting the data into a training set, validation set and test set. Moreover, we discuss how this procedure can be further extended to a k-fold cross-validation, repeated cross-validation, and nested cross-validation. Additionally, we briefly discuss two common approaches for hyperparameter tuning: grid search (an exhaustive search over the search space) and random search (random sampling of hyperparameter settings from the search space).

[**Cost-sensitive Learning**](cost_sensitive_learning)

We explore several approaches to cost-sensitive learning, in which different types of mistakes may be associated with different costs. In particular, we show how a classifier's decision threshold can be optimized to account for different costs.
```

(intromlfurtherreading)=

```{admonition} Further Reading
:class: seealso

* *An Introduction to Machine Learning with Python* by Andreas C. Müller and Sarah Guido. A very practical introduction to machine learning using the popular Python library scikit-learn.

* *Pattern Recognition and Machine Learning* by Christopher Bishop. If you are looking for a deeper understanding of machine learning algorithms, including thorough mathematical details, this book is the way to go. I have heard some people refer to it as the Machine Learning Bible.
```
