(fairml_post_processing)=

# Post-processing Algorithms

Post-processing approaches adjust an existing model, either through post-processing predictions or by adjusting the learned model parameters directly. In this section, we will discuss three approaches: reject option classification, (randomized) group-specific decision thresholds, and leaf relabelling.

(reject_option_classification)=

## Reject Option Classification

Reject option classification is a technique traditionally applied in algorithmic decision-making, where instances for which the predicted class is uncertain are not automatically classified but processed via a different process {footcite:p}`bartlett2008classification`. For instance, rejected instances may be inspected by a human domain expert. {footcite:t}`kamiran2012decision` leverage the principle of reject option classification in the context of fairness. Specifically, the authors assume that more ambiguous instances are most likely to be influenced by biases. Following this assumption, instances for which the classifier is uncertain (i.e., instances close to the decision boundary) are reclassified to promote demographic parity, while maintaining a reasonable level of accuracy.

The reject option classification technique re-classifies instances that fall within what the authors refer to as the _critical region_: $max(P(\hat{Y}=1 \mid x), 1 - P(\hat{Y}=1 \mid x) \leq \theta)$ ({numref}`criticalregion`).

```{figure} ../../figures/criticalregion.svg
:name: criticalregion
:width: 300px
:align: center
:target: none

Reject option classification classifies instances that fall within the critical region with an alternative decision rule, such that the selection rate increases for members of the sensitive group with a low base rate ($x(A)=a$) and decreases for memebers of the sensitive group with a high base rate ($x(A)=a'$).
```

Parameter $\theta$ controls the size of the critical region and can take values $0.5 < \theta < 1$. Instances within the critical region that belong to a sensitive group with a low base rate are classified as positive, while instances within the critical region that belong to a group with a high base rate are classified as negative. Instances outside the critical region are classified according to the standard decision rule for classification that assigns each instance the most likely class.

The standard decision rule, in which each instance is classified according to the most likely class, minimizes the expected loss of classification. Reject option classification diverts from this decision rule by reclassifying instances in the critical region, which generally will increase the expected loss.

Parameter $\theta$ controls the trade-off between accuracy and demographic parity difference: increasing $\theta$ will generally decrease the demographic parity difference as well as the accuracy of the classifier. The value of $\theta$ can be determined based on a validation dataset or as input by a domain expert.

```{tip}
Reject option classification can be interpreted as a cost-based prediction method in which the cost of misclassifying an instance that belongs to a sensitive group affected by historical bias as negative is $\theta/(1−\theta)$ times that of misclassifying it as positive.
```

The reject option classification decision rules can be described as follows.

```{admonition} Reject option classification

{footcite:t}`kamiran2012decision` restrict their approach to a binary classification problem with target variable $Y$, where a positive prediction ($\hat{Y}=1$) corresponds to the desirable output for data subjects, i.e., receiving a benefit or avoiding a burden.

Let $A=a$ be the group with a low base rate and $A=a'$ be the group with a high base rate (i.e., $P(Y=1 \mid A = a) < P(Y=1 \mid A = a')$).

At prediction time, the input of the algorithm is sensitive group membership $x(A)$, predicted score $x(R) = P(\hat{Y}=1 \mid x)$, and parameter $\theta$. An instance is classified as follows:

$$
\hat{Y} =
\begin{cases}
0, & \text{if} x(R) < 1 - \theta \\
0, & \text{if} (x(A)=a') \wedge (\max(r(X), 1 - R(x)) \leq \theta) \\
1, & \text{if} (x(A)=a) \wedge (\max(r(X), 1 - R(x)) \leq \theta) \\
1, & \text{if} x(R) > \theta
\end{cases}
$$

```

Let's consider the toy dataset displayed in {numref}`toydatasetroc`.

```{table} Toy dataset to exemplify the reject option classification technique. $x(R) = P(Y=1 \mid x)$ represents the confidence score assigned by machine learning model.
:name: toydatasetroc

|     | Nationality  | Highest Degree | Job Type   |  $Y$  | $x(R)$    |
| --- | ------------ | -------------- | ---------- | :-:   | :----: |
| 1   | citizen      | university     | board      |  1    |  .99   |
| 2   | citizen      | high school    | board      |  1    |  .90   |
| 3   | citizen      | university     | education  |  1    |  .92   |
| 4   | noncitizen   | university     | healthcare |  1    |  .72   |
| 5   | noncitizen   | none           | healthcare |  0    |  .44   |
| 6   | noncitizen   | high school    | board      |  0    |  .09   |
| 7   | citizen      | university     | education  |  0    |  .66   |
| 8   | citizen      | none           | healthcare |  1    |  .66   |
| 9   | noncitizen   | high school    | education  |  0    |  .32   |
| 10  | citizen      | university     | board      |  1    |  .92   |

```

Assuming `nationality` is considered a sensitive feature, a the default decision threshold is 0.5, a positive prediction (i.e., $\hat{Y}=1$) corresponds to a benefit, we can determine which instances would be re-classified with $\theta=0.7$ as follows.

First, we need to determine which group has a low base rate and which one has a high base rate. $P(Y=1 \mid A=citizen) = 5/6$ while $P(Y=1 \mid A=noncitizen) = 1/4$.

Now, with $\theta=0.7$, we know that all instances with $x(R)$ between 0.3 and 0.7 fall within the critical region and will therefore be reclassified to promote demographic parity. In the toy dataset, instances 5, 7, 8, and 9 fall within the critical region. Instances that do not fall within the critical region (1, 2, 3, 4, 6, and 10) are classified as usual.

In the toy dataset, this implies the following classifications:

- Instance 6 has $x(R) < 0.3$ and will be classified as $\hat{Y} = 0$ .
- Instances 7 and 8 are members of the sensitive group with a high base rate (`nationality=citizen`) and will therefore be classified as $\hat{Y}=0$.
- Instances 5 and 9 are members of the sensitive group with a low base rate (`nationality=noncitizen`) and will therefore be classified as $\hat{Y}=1$.
- Instances 1, 2, 3, 4, and 10 have $x(R) > 0.7$ and will be classified as $\hat{Y} = 1$.

If we had used the default decision rule with a decision threshold of 0.5, the selection rates would have been $P(\hat{Y}=1 \mid A=citizen)=6/6$ and $P(\hat{Y}=1 \mid A=noncitizen)=1/4$. The selection rates after reject option classification are $P(\hat{Y}=1 \mid A=citizen)=4/6$ and $P(\hat{Y}=1 \mid A=noncitizen)=3/4$.

(randomized_thresholds)=

## Randomized Group-Specific Decision Thresholds

If base rates differ across groups, it will often not be possible to identify one unique decision threshold $t$ such that a fairness constraint holds across all groups. In that case, {footcite:t}`hardt2016equality` propose to choose separate thresholds $t_a$ for sensitive groups $a \in A$. For example, in order to satisfy [demographic parity](demographic_parity), we could decrease the decision threshold for a group with a low selection rate, such that more instances are classified as positive. Similarly, considering [equalized odds](equalized_odds), increasing or decreasing a group-specific decision threshold allows us to control the trade-off between false positives and false negatives for each sensitive group separately.

Group-specific thresholds are not always sufficient to achieve equalized odds. The trade-off between false positives and false negatives is often analyzed using a [Receiver Operating Characteristic (ROC) curve](roc_curve), which sets out a classifier's false positive rate against its true positive rate over varying decision thresholds. Considering equalized odds, group-specific thresholds limit us to the combinations of error rates that lie on the _intersection_ of the group-specific ROC curves.

In some cases, the group-specific ROC curves may not intersect or represent a poor trade-off between false positives and false negatives. To further increase the solution space, {footcite:t}`hardt2016equality` allow the decision thresholds to be randomized. That is, the decision threshold $T_a$ is a randomized mixture of two decision thresholds $\underline{t}_{a}$ and $\overline{t}_{a}$ ({numref}`randomizedthresholds`).

```{figure} ../../figures/randomizedthresholds.svg
:name: randomizedthresholds
:width: 300px
:align: center
:target: none

A randomized decision threshold is a randomized mixture of two decision thresholds: a random choice between two decision thresholds $\underline{t}_{a}$ and $\overline{t}_{a}$ with probability $p$.
```

Randomization allows us to achieve _any_ combination of error rates that lies within the convex hull of the ROC curve. In cases where group-specific ROC curves do not intersect apart from trivial end points, the predictive performance of the model for the best-off group can be artificially lowered through randomization until the performance is equal to that of the worst-off group ({numref}`randomizedthresholdsroc`).

```{figure} ../../figures/randomizedthresholdsroc.svg
:name: randomizedthresholdsroc
:width: 300px
:align: center
:target: none

Randomization of the decision threshold between values $\underline{t}_{a}$ and $\overline{t}_{a}$ allows us to achieve any ($fpr$, $tpr$) combination on the line segment between the corresponding points on the ROC curve. The exact coordinate is determined by $p_a$, where higher values of $p_a$ are closer to $\overline{t}_{a}$ and vice versa. In this case, the group-specific ROC curves do not intersect, so we cannot find non-randomized group-specific decision thresholds such that equalized odds is satisfied. Randomizing the decision threshold for Group 1 increases the solution space such that equalized odds can be achieved.
```

In cases where the ROC curves do intersect, but at a sub-optimal point, randomization allows us to achieve a more favorable point on the ROC curve. Note that the maximum true positive rate and true negative rate are always limited by the model's performance for the sensitive group for which the model performs the worst.

(leaf_relabeling)=

## Leaf Relabelling

In addition to post-processing of predictions, some fairness-aware machine learning algorithms directly adjust the parameters of a trained model to enforce a fairness constraint. One of the earliest of these approaches is the leaf relabelling algorithm for decision trees introduced by {footcite:t}`kamiran2010discrimination`.

The default decision rule of a decision tree classifies new instances based on the majority class of the training instances that reside in each leaf. When base rates are unequal in the training data, a decision tree will likely reproduce this disparity. To ensure selection rates are equal, we can control the group-specific selection rates by relabeling one or more leaves of the decision tree. Relabeling a leaf from negative to positive increases the selection rate of a sensitive group whose members end up in that leaf relatively often. Vice versa, relabeling a leaf from positive to negative decreases the selection rate of a highly represented sensitive group.

The leaf relabeling algorithm is designed to swap the predicted class of leaves in the decision tree to satisfy a demographic parity difference constraint while maintaining accuracy as much as possible. The algorithm greedily relabels leaves with the largest decrease in demographic parity difference and the lowest decrease in accuracy, until the demographic parity difference is smaller than a predetermined threshold $\epsilon$.

```{admonition} Leaf Relabeling
The leaf relabeling algorithm changes the predicted class of selected leaves to satisfy a demographic parity difference constraint while maintaining accuracy as much as possible.

The algorithm takes as input decision tree $T$.

1. _Compute the influence of relabeling._ For each leaf in the tree $l \in T$
    * $\Delta acc_l$ : change in accuracy if leaf $l$ is relabeled.
    * $\Delta dpd_l$ : change in demographic parity difference if leaf $l$ is relabeled.
2. _Greedy relabeling._
    * Leaves to relabel: $L = \{ \}$.
    * $dpd_T$ : the demographic parity difference of the decision tree.
    * Candidate leaves $I = \{ l \in L \mid \Delta dpd_l < 0\}$.
    * While $dpd_T > \epsilon$:
        * Identify the best remaining leaf for relabeling: $\underset{{i \in I \setminus L} }{\operatorname{argmax}} (\Delta dpd_l / \Delta acc_l)$
        * Update $L = L \cup \{l\}$
        * Update $dpd_T$
```

Let's look at an example. Consider the decision tree in {numref}`leafrelabeling`. The majority of training instances in leaf $l_3$ are negatives (2 out of 3). The default decision rule for decision trees would thus classify new instances that fall in leaf $l_3$ as negative. What would be the effect of changing the label of leaf $l_3$ from negative to positive?

```{figure} ../../figures/leafrelabeling.svg
:name: leafrelabeling
:width: 400px
:align: center
:target: none

A decision tree trained on a data set containing two categorical features, feature 1 and feature 2. Each rectangle represents a leaf of the decision tree. Each dot represents an instance in the training data set. The default decision rule of a decision tree classifies new instances based on the majority class of the training instances that reside in each leaf.

```

First, we consider the change in accuracy. We have 20 instances. With the original label, leaf $l_3$ classifies 2 instances correctly as negative and 1 instance incorrectly as negative. If we were to change the label from negative to positive, we classify 2 negative instances incorrectly as positive and 1 positive instance correctly. The difference in accuracy between the original tree and a tree where leaf $l_3$ is relabeled is therefore: $\Delta acc_l = (1 - 2)/20 = -1/20$.

Second, consider the change in demographic parity difference. Each sensitive group has 10 instances. After relabeling, 1 instance of group 0 will be classified as positive, rather than negative, increasing the selection rate by 1/10. At the same time, 2 instances of group 1 are reclassified from negative to positive, increasing the selection rate by 2/10. The change in demographic parity difference is,therefore, $\Delta dpd_l = 1/10 - 2/10 = -1/10$.

## References

```{footbibliography}

```
