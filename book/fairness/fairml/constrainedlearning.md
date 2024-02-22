(fairml_constrained_learning)=

# Constrained Learning Algorithms

Constrained learning techniques directly incorporate a fairness constraint during the training of a machine learning model. Most techniques achieve this by adjusting an existing machine learning paradigm to incorporate a fairness constraint directly into the objective function of the machine learning algorithm. This typically involves the adjustment of the loss function to penalize unfair outcomes, according to the fairness metric of choice. Other techniques impose a fairness constraint through meta-estimators: a wrapper around a regular machine learning algorithm, typically resulting in an ensemble of classifiers. In this section, we will discuss four constrained learning approaches: [reweighing](reweighing), [regularization](regularization), [decoupled classification](decoupled_classification), and [adversarial learning](adversarial_learning).

```{note}
The goal of this section is **not** to present a comprehensive overview of the fairness-aware machine learning techniques proposed in the algorithmic fairness literature. Instead, these examples allow you to develop an intuition of the workings, (dis)advantages, and limitations of the fairness-aware machine learning paradigm.
```

(reweighing)=

## Reweighing

[Cost-sensitive learning](cost_sensitive_learning) is a set of learning techniques commonly used to explicitly take into consideration different costs of misclassification during model training and selection. In particular, many machine learning algorithms allow assigning different weights to instances in the algorithm's loss function, such that the learning algorithm is penalized more for misclassifying an instance with a high weight compared to an instance with a low weight.

{footcite:t}`kamiran2012data` leverage cost-sensitive learning by reweighing instances in the data set in such a way that it reduces the association between a sensitive feature and the target variable. Instead of assigning general class weights, as is common in cost-sensitive learning for imbalanced classification problems, the reweighing technique introduces a weighting scheme that discourages an association between sensitive feature $A$ and the classifier's predictions $\hat{Y}$.

The intuition behind the reweighing scheme is as follows. In a dataset unaffected by [historical bias](historical_bias), sensitive feature $A$ and target variable $Y$ are statistically independent. As such, the expected probability of an instance belonging to sensitive group $A=a$ and the positive class $Y=1$ can be determined by multiplying the probabilities of sensitive group membership and belonging to the positive class: $P_{exp}(Y=1 \wedge A=a) = P(Y=1) \times P(A=a)$. In a dataset affected by historical bias, the observed probability $P_{obs}(Y=1 \wedge A=a)$ will not be equal to $P_{exp}$. To compensate for this, the reweighing technique assigns weights to instances according to the observed and expected probabilities. Subgroups with a lower probability than expected in the absence of historical bias will be assigned a high weight, while subgroups with a higher probability than expected will be assigned a lower weight.

```{admonition} Reweighing

{footcite:t}`kamiran2012data` restrict the approach to a single binary sensitive attribute $A$ and a binary classification problem with target variable $Y$, where a positive prediction ($\hat{Y}=1$) corresponds to the desirable output for data subjects, i.e., receiving a benefit or avoiding a burden.

The input of the algorithm is labeled dataset $D = \{X, Y, A\}$. Let $x(A)$ be a function that retrieves the value of $A$ for instance $x$.

1. Assuming $A$ can take the values $a$ and $a'$, and $Y$ is either $0$ or $1$, we partition the training data into four subgroups:
    * $D_{a, 1} := \{(x,y) \in D \mid x(A) = a \wedge x(Y) = 1 \}$
    * $D_{a, 0} := \{(x,y) \in D \mid x(A) = a \wedge x(Y) = 0 \}$
    * $D_{a', 1} := \{(x,y) \in D \mid x(A) = a' \wedge x(Y) = 1 \}$
    * $D_{a', 0} :=  \{(x,y) \in D \mid x(A) = a' \wedge x(Y) = 0 \}$
2. Compute weights for each subgroup: $W_{a,y} = \frac{P_{{a,y}}^{exp}}{P_{{a,y}}^{obs}}$. Let $N := |D|$ (i.e., be the total number of instances in the training data). Then, for each subgroup $D_{a,y}$, we have:
    * $P_{a,y}^{exp} = P(A=a) \times P(Y=y) = \frac{|\{ x \ in D \mid x(A) = a \}|}{N} \times \frac{|\{ x \ in D \mid x(Y) = y \}|}{N}$
    * $P_{{a,y}}^{obs} = P(A=a \wedge Y=y) = \frac{|\{ x \ in D \mid x(A) = a, x(Y) =y \}|}{N} = \frac{|D_{a,y}|}{N}$
3. Learn a classifier, taking into consideration the subgroup weights $W_{a,y}$ for all values of $a \in A$ and $y \in Y$.

```

We can illustrate the reweighing technique with an example. Consider the toy data set displayed in {numref}`toydatasetreweighing`. Assuming `nationality` is considered a sensitive feature and a positive prediction (i.e., $\hat{Y}=1$) corresponds to a benefit, we can determine subgroup weights as follows.

```{table} Toy dataset to exemplify the reweighing technique.
:name: toydatasetreweighing

|     | Nationality  | Highest Degree | Job Type   |  $Y$  |
| --- | ------------ | -------------- | ---------- | :-:   |
| 1   | citizen      | university     | board      |  1    |
| 2   | citizen      | high school    | board      |  1    |
| 3   | citizen      | university     | education  |  1    |
| 4   | noncitizen   | university     | healthcare |  1    |
| 5   | noncitizen   | none           | healthcare |  0    |
| 6   | noncitizen   | high school    | board      |  0    |
| 7   | citizen      | university     | education  |  0    |
| 8   | citizen      | none           | healthcare |  1    |
| 9   | noncitizen   | high school    | education  |  0    |
| 10  | citizen      | university     | board      |  1    |

```

During training, each instance will be weighted by a weight determined via subgroup membership defined by the sensitive group and target variable. For example, consider instance 4. This instance is registered as `Nationality=noncitizen` and $Y=1$. To determine its weight, we need to consider the expected and observed probability of the subgroup it belongs to.

Out of all 10 instances, instance 4 is the only instance corresponding to this combination of sensitive group membership and class. Therefore, the observed probability $P_{noncitizen, 1}^{obs} = 1/10$. The expected probability that would have occurred in the absence of an association between the sensitive feature and target variable can be computed by considering the probabilities of `Nationality=noncitizen` and $Y=1$ in the full data set: $P_{noncitizen, 1}^{exp} = (4/10) \times (6/10) = 0.24$. Finally, the weight for instance 4 is equal to $W_{noncitizen, 1} = \frac{0.24}{0.1} = 2.4$.

(regularization)=

## Regularization

Regularization is a common approach in machine learning to avoid the problem of [overfitting](model_selection). A regularization term is added to the loss function of the algorithm to penalize more complex solutions. In the case of logistic regression, the complexity is quantified in terms of the magnitude of the learned coefficients $\beta_j \in \Theta$, with $j \in \{1 ..., k\}$. In $L_1$ regularization, also known as lasso regression, the absolute value of the magnitude of the coefficients is penalized: $\sum_{j=1}^k |\beta_{j}|$. In this way, the algorithm is encouraged to reduce coefficients to zero, resulting in a sparser model and (implicit) feature selection. In $L_2$ regularization, also known as ridge regression, the algorithm is encouraged to shrink coefficients to zero - but not to entirely reduce them. This is achieved via a penalty term that considers the squared magnitude of the coefficients: $\sum_{j=1}^k \beta_{j}^2$.

Several researchers have proposed to add another type of regularization to the loss function of the machine learning objective that is designed to penalize solutions that deviate from a fairness constraint. For example, {footcite:t}`kamishima2012fairness` propose a logistic regression algorithm that employs a fairness regularizer designed to minimize the demographic parity difference.

Specifically, given a training dataset $D = (X,A,Y)$ and regression coefficients $\Theta$, the authors propose to use the following loss function:

$$-L(D, \Theta) + \frac{\lambda}{2} |\Theta|_2^2 + \eta R(D, \Theta)$$

The first term, $-L(D, \Theta)$ corresponds to the (negative) maximum log-likelihood, which is the standard optimization objective of a logistic regression classifier. The second term, $\frac{\lambda}{2} |\Theta|_2^2$ corresponds to the standard $L_2$ regularization term to avoid overfitting, accompanied by hyperparameter $\lambda$ which controls the amount of regularization. The third term, $\eta R(D, \Theta)$, introduces the fairness regularizer:

$$ R(D, \Theta) = \sum*{(x, a) \in D} \sum*{y \in \{0, 1 \}} M[y \mid x, a, \Theta] \ln \frac{\hat{P}(y \mid a)}{\hat{P}(y)} $$

where $\hat{P}(y)$ and $\hat{P}(y \mid a)$ correspond to approximations of the probability an instance belongs to class $y$ and the probability an instance belongs to class $y$ given sensitive group membership $a$ respectively, and $M[y \mid x, a, \Theta]$ corresponds to the conditional probability of a class given the features, sensitive features, and model parameters.

It suffices to know that $R(D, \Theta)$ becomes large when the approximate mutual information - a measure of association - between the predictions of the classifier and the sensitive feature is high. Similar to standard $l_2$ regularization, the fairness regularizer is accompanied by a hyperparameter $\eta$ that controls the amount of regularization. Larger values of $\lambda$ and $\eta$ correspond to higher amounts of regularization, and vice versa.

(decoupled_classification)=

## Decoupled Classification

The regularization approach described above is designed specifically for logistic regression models. In some cases, it might be difficult to fit a single model that performs well for all sensitive groups. To overcome this problem, researchers have proposed various meta-estimators: a wrapper around a standard machine learning algorithm, where fairness constraints are imposed through careful selection of an ensemble of classifiers.

{footcite:t}`dwork2018decoupled` propose a decoupled classification system that acts as such a meta-estimator. The main procedure consists of two steps: (1) learn a set of classifiers for each sensitive group with varying selection rates, (2) from the set of learned classifiers, select one classifier for each group, such that the joint loss, a loss function designed to include both overall predictive performance and a fairness constraint, is minimized.

The input of decoupled classification is a $C$-learning algorithm $M:(\mathcal{X},\mathcal{Y}) \rightarrow 2^C$, which returns one or more classifiers from a learning algorithm $C$ with differing numbers of positive classifications on the training data. Note that it is possible to output a classifier with a varying number of positive classifications via [cost-sensitive learning](cost_sensitive_learning) approaches such as decision threshold selection, reweighing, and sampling.

From the set of classifiers generated in this step, exactly one classifier is selected for each sensitive group, resulting in ensemble $\gamma(C) = \{C_1, C_2, ..., C_k\}$, where $K$ is the number of sensitive groups. The ensemble is selected to minimize the joint loss. Provided the joint loss function satisfies a weak form of monotonicity, the decoupled classification approach can identify a decoupled solution that minimizes the joint loss for any off-the-shelf learning algorithm. Various notions of fairness, including [demographic parity](demographic_parity) and [equalized odds](equalized_odds), can be represented in such a joint loss function.

For example, let $L$ correspond to the overall loss and $p_a$ to the faction of positively classified examples among instances of group $a$. Then, the joint loss related to demographic parity can be expressed as:

$$\lambda L + (1-\lambda) \sum_{k=1}^K \mid p_k - \frac{1}{K} \sum_{k'=1}^K p_{k'} \mid $$

where $\lambda$ represents a hyperparameter that controls the relative importance of demographic parity over the overall loss.

```{admonition} Decoupled Classification

The decoupled classification algorithm takes as input a $C$-learning algorithm $M$ and a dataset $D={X,Y,A}$, where $X$ corresponds to features, $Y$ the target variable, and $A$ a sensitive feature. Let $x(A)$ be a function that retrieves the value of $A$ for instance $x$.

1. Partition the data by sensitive group and run the learning algorithm on each group: for all $a \in A$, learn a classifier $C_a = M(\{(x,y) \in D \mid x(A)=a\})$. Within each group, the learner outputs one or more classiffers of differing numbers of positives.
2. Return $\gamma(C)$, a set of classifiers selected from the classifiers generated in the previous step, such that the joint loss is minimized.
```

A potential problem is that the number of instances that belong to a particular sensitive group may be insufficient to learn an accurate classifier for that group. In addition to the simple procedure described above, the authors therefore also introduce a simple transfer learning algorithm that allows taking into consideration out-group instances, down-weighted compared to in-group instances.

(adversarial_learning)=

## Adversarial Learning

Another approach that can be used to introduce fairness constraints during training is adversarial learning. Adversarial learning was first introduced in the context of Generative Adversarial Networks (GAN) for image generation {footcite:p}`goodfellow2014generative`.
A detailed description of GANs is outside of the scope of these lecture notes. It suffices to know that GANs are a set of deep learning frameworks consisting of two neural networks that compete in a zero-sum game. In image generation, for example, one model learns to generate new data, while the other network predicts whether the generated data belongs to the original data set or not. The first network is penalized if the second network can accurately discern whether the image was generated by the first network or was sampled from the original data distribution.

{footcite:t}`zhang2018mitigating` introduce a fairness-aware machine learning approach that leverages an adversarial learning framework to enforce a fairness constraint. The framework consists of two neural networks: a predictor model, which is designed to accurately predict the target variable, and an adversarial model, which is designed to predict the violation of a fairness constraint ({numref}`adversariallearning`). The loss of the adversarial model $L_{adv}(\hat{a},a)$ is used to penalize the loss of the predictor model, $L_{pred}(\hat{y},y)$, such that the predictor is encouraged to ensure the adversarial model performs poorly.

```{figure} ../../figures/adversariallearning.svg
:name: adversariallearning
:width: 500px
:align: center
:target: none

The adversarial learning technique consists of two neural networks: a predictor model, which is designed to minimize the standard loss $L_{pred}(\hat{Y},Y)$ and an adversarial model, which is designed to minimize a loss function related to violation of a fairness constraint, $L_{adv}(\hat{a},a)$.
```

The authors introduce two variations of the framework, that optimize for either [demographic parity](demographic_parity) or [equalized odds](equalized_odds). When the goal is to achieve demographic parity, the adversary is designed to predict sensitive feature $A$ based on the predicted probabilities $\hat{Y}$ of the predictor model. When the goal is to achieve equalized odds, the adversary must predict the sensitive feature $A$, taking as input both the predicted probabilities $\hat{Y}$ and the ground-truth variable $Y$.

## References

```{footbibliography}

```

$$
$$
