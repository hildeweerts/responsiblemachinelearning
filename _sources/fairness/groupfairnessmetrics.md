---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(group_fairness_metrics)=
# Group Fairness Metrics
There have been many attempts to formalize fairness in mathematical criteria, but by far the most prominent notion of fairness is {term}`group fairness`. Group fairness metrics measure the extent to which a particular *group statistic* differs across (sub)groups of individuals. In technical literature, these groups are typically referred to as {term}`sensitive group`s and are often - *but not always\* - defined based on legally protected characteristics such as race and gender.

Group fairness metrics differ primarily based on *which* group statistic is required to be equal across groups. We can distinguish three main categories: metrics based on [independence](independence_metrics), [misclassification rates](misclassification_metrics), and [calibration](calibration_metrics). 

In the remainder of this section, we formally define several group fairness criteria in the classification scenario and illustrate how they can be computed using the Python library `fairlearn`. 

Throughout this section, we use the following notation:

* $X$ : the set of input features;
* $Y$ : the 'ground truth' labels;
* $\widehat{Y}$ : the predictions;
* $A$ : the sensitive feature(s) that measure sensitive characteristic(s).

## Running Example
As a running example, we train a logistic regression model on a synthetic hiring data set. 

The data set has a few features that can be used for predictions, including `years_of_experience`, the `test_score` of an applicant on a job-related test, the `interview_score` that is provided by person who interviewed the candidate, the `average_grade` the applicant achieved in their highest degree, and finally the applicant's `gender`. The target variable is `hired`.

```{code-cell} ipython3
---
tags: [hide-input]
---

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# load data
data = pd.read_csv('hiring.csv')
# display data set
display(data.head())

# prepare data set
y = data['hired']
A = data['gender']
X = data.drop(['hired', 'gender'], axis=1)

# convert categorical gender to dummy variables
X[['female', 'male', 'non_binary']] = pd.get_dummies(A)

# split into train and test set
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A, random_state=0, test_size=0.5)

# train classifier
lr = LogisticRegression()
lr.fit(X_train, y_train)

```

```{caution}

The data that will be used throughout this section is **synthetic** and is merely used for illustrative purposes. **I do *not* claim in any way that this data set and model are representative of actual hiring practices**.

```

(independence_metrics)=
## Independence
When we think of discrimination, an intuitive notion of fairness is independence: whether an individual receives a particular output, should not depend on sensitive group membership. For example, in our hiring scenario, one may argue that whether an applicant is hired should not depend on their gender.

### Demographic parity
The first criterion that we will discuss is {term}`demographic parity`. In a classification scenario, {term}`demographic parity` requires that, for all values of $y$ and $a$:

$$P(\hat{Y} = y \mid A = a) = P(\hat{Y} = y \mid A = a')$$

In other words, the probability of receiving a particular prediction should be **independent** of sensitive group membership. In a binary classification scenario, this criterion can be checked by comparing {term}`selection rate`s, i.e., the proportion of predicted positives, across groups. When positive predictions correspond to a resource that is distributed between people, {term}`demographic parity` can be interpreted as a measure of potential {term}`allocation harm`.

In many cases, we might be interested to measure not just whether demographic parity holds exactly, but also *to what extent* it is violated. A popular way to summarize the comparison of group statistics is to compute the maximum *difference* between sensitive groups: 

$$\max_{a, a' \in A}( P(\hat{Y} = 1 \mid A = a) - P(\hat{Y} = 1 \mid A = a') )$$

Alternatively, we can compute the minimum *ratio* between groups:

$$\min_{a, a' \in A}(P(\hat{Y} = 1 \mid A = a) / P(\hat{Y} = 1 \mid A = a'))$$

Note that demographic parity depends on predictions $\hat{Y}$ and sensitive feature(s) $A$, but **not** on the 'ground-truth' target variable $Y$. A direct consequence of this fact is that if group-specific {term}`base rate`s $p_a = P(Y=1 \mid A=a)$ (i.e., the proportion of positives) differs between groups **a perfect classifier cannot satisfy demographic parity**. We will explore normative arguments for and against demographic parity in more detail in [Choosing the 'right' fairness metric](normative_underpinnings).

```{tip} 
Demographic parity as a fairness metric is *very loosely* inspired by the legal concepts of indirect discrimination in EU law and disparate impact in US labor law - but satisfying the metric definitely does not directly imply legal compliance!

```

In `fairlearn`, we can easily compute group-specific statistics and subsequent aggregations using {class}`fairlearn.metrics.MetricFrame`.

```{code-cell} ipython3
# compute metrics by group
from fairlearn.metrics import MetricFrame, selection_rate

mf = MetricFrame(
    metrics={"selection rate": selection_rate},
    y_true=y_test,  # is ignored
    y_pred=lr.predict(X_test),
    sensitive_features=A_test,
)

# print results
display(mf.by_group)
print("Overall selection rate: %.2f" % mf.overall)

# compute demographic parity as the max difference between groups
print("demographic parity difference: %.2f" % mf.difference(method="between_groups"))

# compute demographic parity as the min ratio between groups
print("demographic parity ratio: %.2f" % mf.ratio(method="between_groups"))
```

From these results, we can see that our model has a higher selection rate for `male` applicants compared to other applicants, particularly `non_binary` applicants. Without further context, we cannot conclude whether this disparity is problematic. However, the low selection rates of some group can hint towards potential {term}`allocation harm` and calls for further investigation.

```{tip} 
You can also directly compute the extent to which demographic parity is violated using
{py:func}`fairlearn.metrics.demographic_parity_difference` or {py:func}`fairlearn.metrics.demographic_parity_ratio`. These summary metrics are particularly convenient for hyperparameter tuning or monitoring. 

For exploratory and manual inspections, I highly recommend sticking to {py:class}`fairlearn.metrics.MetricFrame`, as this allows you to make more detailed comparisons across groups. While the demographic parity difference indicates the (maximum) disparity, it does not tell you between which groups the disparity occurred, nor what the selection rates of those groups are.
```

### Conditional Demographic Parity
Apart from employment, there may be characteristics that, either from a legal or ethical perspective, legitimize differences between groups. Loosely inspired by these legal imperatives, {footcite:t}`Kamiran2013` put forward a notion of fairness that we will refer to as {term}`conditional demographic parity`. This is a variant of demographic parity that allows for differences between groups, if these differences are justified, from a legal or ethical point of view, by a {term}`control feature`. 

Conditional group fairness is best illustrated by an example. Imagine a scenario in which women have a lower income, on average, than men. This may imply that women are treated unfairly. However, what if many women work fewer hours than men? In this case, the observed disparity can be (at least partly) explained by the lower number of working hours. Consequently, equalizing income between men and women would mean that women are paid more per hour than men. If we believe unequal hourly wages to be unfair, we can instead equalize income only between women and men who work similar hours. In other words, we minimize the difference that is still present *after* controlling for working hours. 

Formally, let $W$ be a control feature. Then, conditional demographic parity holds if, for all values of $y$, $a$, and $w$:

$$P(\hat{Y} = y \mid A = a, W=w) = P(\hat{Y} = y \mid A = a', W=w)$$

Conditional demographic parity is particularly relevant considering Simpson's paradox. This paradox states that if a correlation occurs in several different groups, it may disappear or even reverse when the groups are aggregated.

```{admonition} *Example:* Simpson's Paradox: Berkeley University Admissions
When considering all programs together, women were accepted less often than men, implying a gender bias. However, it turned out that women at Berkeley often apply for competitive programs with a relatively low acceptance rate. As a result, the overall acceptance rate of women in the aggregated data was lower -- even though the acceptance rate of women *within* each program was higher than the acceptance rate of men. Hence, if the admission's office would have tried to equalize the overall acceptance rate between men and women, men would have received an even lower acceptance rate.
```

In `fairlearn`, the `control_features` parameter allows you to compare group statistics across the values of a feature we wish to control for. 

When we make comparisons across multiple variables (or variables with a high number of categories), the number of instances within a group can become very small. Small sample sizes can be problematic, as the group statistic estimates become less reliable. We can use {py:func}`fairlearn.metrics.count` to inspect the number of instances in a {py:class}`fairlearn.metrics.MetricFrame` along with the metrics of interest.

Let's see whether the disparity in selection rates can be explained by the `test_score` a participant achieved.

```{code-cell} ipython3
from fairlearn.metrics import count

# compute metrics by group
mf = MetricFrame(
    metrics={
        "selection rate": selection_rate, 
        "count": count},
    y_true=y_test,  # is ignored
    y_pred=lr.predict(X_test),
    sensitive_features=A_test,
    control_features=X_test['test_score']
)

# display results
display(mf.by_group)
display(mf.difference())
``` 

Interestingly, the disparity in selection rates is much lower for participants with a low `test_score`. However, it is particularly high for participants with a very high `test_score`. As such, `test_score` does not adequately explain disparities.

However, note that the number of instances in some of the subgroups have become very small. In particular, there are only 22 instances for which `gender=non_binary` *and* `test_score=3`. As such, the computed selection rate for this group is much less reliable - in practice, it may turn out to be much higher or lower than what we observe here.

```{note}
**Small sample sizes are a very common issue in fairness assessments** - particularly when we consider intersectional notions of fairness that consider multiple sensitive characteristics or cross-validation procedures that further split the data set into smaller portions. Sample size is an important factor to consider before you do any interventions based on your findings, as the estimations of group statistics can be very uncertain. 

The issue of small sample sizes is further complicated by the multiple comparison problem. If we want to statistically test whether a metric is significantly different for one group compared to other groups, we need to make many comparisons between groups. As the number of groups increases, the probability of making a wrong inference increases as well, requiring a stricter significance threshold for each individual comparison. When sample sizes are small, it becomes hard to draw any conclusions at all.

In some cases, it might be possible to collect more data, but statistically sound fairness assessments are still an active area of research.
```

(misclassification_metrics)=
## Misclassification Rates
The second set of fairness metrics that we will consider are related to misclassification rates.

### Equalized Odds
{term}`Equalized odds` {footcite:p}`hardt2016equality` is one of the most commonly studied fairness metrics. It requires an equal distribution of classification errors across sensitive groups for all classes. Formally, equalized odds is satisfied if for all values of $y \in Y$ and $a, a' \in A$, 

$$P(\hat{Y} = y \mid A = a, Y = y) = P(\hat{Y} = y \mid A = a', Y = y)$$ 

In a binary classification scenario, satisfying equalized odds boils down to equal group-specific {term}`false positive rate`s and {term}`false negative rate`s (or, equivalently, equal {term}`true negative rate`s and {term}`true positive rate`s). Essenstially, equalized odds requires that, given a particular 'ground-truth' label, the probability of receiving a predicted label is independent of sensitive group membership. For example, in our hiring scenario, a classifier that satisfies equalized odds is neither more likely to falsely *reject* suitable `male` candidates compared `non_binary` or `female` candidates, nor is it more likely to falsely accept unsuitable candidates for some genders. As such, we can interpret equalized odds as one way to measure the risk of {term}`allocation harm` or {term}`quality-of-service harm`.


```{tip} 

If you need a refresher on different evaluation metrics for classification models, check out the [Model Evaluation](model_evaluation_classification_metrics) section of the machine learning preliminaries.

```

Similar to the demographic parity difference, we can summarize the comparisons of the false positive rate and false negative rates as the maximum absolute difference between groups. We can further summarize equalized odds as the maximum between the false positive rate difference and false negative rate differnece. Formally, that is:

$$\max_{a, a' \in A, y \in Y}( \mid P(\hat{Y} = y \mid A = a, Y = y) - P(\hat{Y} = y \mid A = a', Y = y) \mid )$$

Alternatively, one can take the average over the maximum difference.

In `fairlearn`, we can compare the false positive rates and false negative rates across groups in a single {py:class}`fairlearn.metrics.MetricFrame`. Additionally, we can use {py:func}`fairlearn.metrics.equalized_odds_difference` to directly compute the maximum of the false positive rate difference and false negative rate difference. 

```{code-cell} ipython3
from fairlearn.metrics import false_positive_rate, false_negative_rate, equalized_odds_difference

# compute metrics by group
mf = MetricFrame(
    metrics={
        'fpr' : false_positive_rate,
        'fnr' : false_negative_rate,
        },
    y_true=y_test,
    y_pred=lr.predict(X_test),
    sensitive_features=A_test
)

# display results
display(mf.by_group)
display(mf.difference())

# compute equalized odds difference directly
eod = equalized_odds_difference(
    y_true=y_test,
    y_pred=lr.predict(X_test),
    sensitive_features=A_test
)
print("equalized odds difference: %.2f" % eod)
```

From these results we can conclude that the group-specific false negative rates differ a lot, while the false positive rates are relatively similar. This implies that the model more often falsely predicts `female` and especially `non_binary` applicants to be rejected compared to `male` applicants.

### Equal Opportunity
Equal opportunity is a relaxation of equalized odds that only requires the true positive rate (or, equivalently, false negative rate) to be equal across groups. Formally, for all values of $a, a' \in A$:

$$P(\hat{Y} = 1 \mid A = a, Y = 1) = P(\hat{Y} = 1 \mid A = a', Y = 1)$$ 

```{tip}

Equal opportunity as a fairness metric is *very loosely* inspired by the egalitarian concept of equality of opportunity, which we will discuss in more detail in [Choosing the "right" fairness metric](normative_underpinnings).

```

As we have seen before, the hiring model does not satisfy equal opportunity, as the false negative rates differ between sensitive groups.

(calibration_metrics)=
## Calibration
The calibration of a machine learning model reflects whether predicted probabilities are consistent with observed probabilities. For example, a model is well-calibrated if out of all instances that receive a confidence score of 0.7, the fraction of instances that actually belongs to the positive class is also 0.7.

```{tip}

You can find a refresher on calibration of machine learning models in the machine learning preliminaries section on [model calibration](model_calibration). 

```

### Equal calibration
Calibration is particularly relevant to interpret the output of a predictive model in a risk assessment scenario. In those cases, the output of the machine learning model can be interpreted as a *risk score* $R$. The equal calibration fairness metric requires that the model is equally calibrated for each sensitive group. That is, equal calibration holds if, for all values of $y \in Y$, $a \in A$, and $r \in R$, we have:

$$P(Y = y \mid A = a, \hat{R} = r) = P(Y = y \mid A = a', \hat{R} = r)$$

In other words, for each possible risk score, the probability that you belong to a particular class is the same, regardless of sensitive group membership. For example, given that an instance is predicted to belong to the negative class, the probability of actually belonging to the negative class is independent of sensitive group membership. Essensially, equal calibration requires that the *meaning* of a particular score is the same, regardless of sensitive group membership. 

We can evaluate the calibration of risk scores by plotting group-specific calibration curves.

```{code-cell} ipython3
---
tags: [hide-input]
---

import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay

# display calibration curves
ax = plt.gca()
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

for gender in ['non_binary', 'female', 'male']:
    CalibrationDisplay.from_predictions(
        y_true=y_test[A_test==gender],
        y_prob=lr.predict_proba(X_test[A_test==gender])[:,1],
        n_bins=5,
        ax=ax, 
        label=gender)
plt.show()
```

From this plot, we can see that the model is similarly calibrated for each of the `gender` categories.

In the binary classification scenario, equal calibration implies that the {term}`positive predictive value` (which is equivalent to {term}`precision`) and {term}`negative predictive value` are equal across groups. That is:

$$P(Y = y \mid A = a, \hat{Y} = y) = P(Y = y \mid A = a', \hat{Y} = y)$$

As with equalized odds, equal calibration in the binary classification scenario can be summarized as the maximum difference in positive predictive value difference and negative predictive value difference:

$$\max_{a, a' \in A, y \in Y}P(Y = y \mid A = a, \hat{Y} = y) - P(Y = y \mid A = a', \hat{Y} = y)$$

```{code-cell} ipython3
from sklearn.metrics import precision_score 

# first, we define a function to compute the negative predictie value
def negative_predictive_value_score(y_true, y_pred, **kwargs):
    """
    NPV is not in scikit-learn, but is the same as PPV but with 0 and 1 swapped.
    """
    return precision_score(y_true, y_pred, pos_label=0, **kwargs)

# compute metrics
mf = MetricFrame(
    metrics={
        "positive predictive value": precision_score,
        "negative predictive value": negative_predictive_value_score,
    },
    y_true=y_test,
    y_pred=lr.predict(X_test),
    sensitive_features=A_test,
)
display(mf.by_group)
```

Again, we can see that the hiring classifier is similarly calibrated, as the group-specific positive and negative predictive values are very close.

```{tip}
In `fairlearn` we can define a custom fairness metric for NPV using {py:func}`fairlearn.metrics.make_derived_metric`. This function takes as parameters `metric` (a callable metric such as {py:func}`sklearn.metrics.recall_score` or {py:func}`fairlearn.metrics.false_positive_rate`) and `transform` (a string indicating the type of transformation, e.g., `difference` or `group_min`). The function returns a function with the same signature as the supplied metric, but with additional `sensitive_features=` and `method=` arguments.
```

## Limitations of Group Fairness Metrics
Group fairness metrics as measures of fairness of machine learning models have several benefits. As the criteria are based on statistics that are commonly used to evaluate machine learning models, they are conceptually easy to understand. Moreover, a disaggregated is analysis is straightforward to implement.

However, group fairness metrics also pose several challenges. We already touched upon one of them: the risk of **imprecise fairness assessments** due to small sample sizes and the problem of multiple comparisons. A second problem that is often encountered in practice is a lack of **access to sensitive features** due to privacy regulations or practical availability. Moreover, even **identifying sensitive groups** that are at risk of fairness-related harm (and how to measure sensitive group membership) is a non-trivial task that requires a deep understanding of the context of the system you are developing. Additionally, group fairness metrics **only consider the direct outcome of the model, rather than the outcome of the system of which the model is a part**. For example, the whether a candicate gets hired does not only depend on the predictions of a resume selection model, but als on how recruiters interpret the predictions and what happens in the other steps of the hiring process.

At a more conceptual level, group fairness metrics operationalize fairness as something that is to be achieved *between groups*, which may not translate to the individual level. In particular, **group statistics may disguise differences *within* groups**. Moreover, group fairness metrics implicitly assume that a particular outcome (e.g., receiving a positive prediction or being errenously classified as a negative) is universally beneficial or harmful and thus can **disregard individual experiences** of benefits and harm. Finally, group fairness metrics are **merely observational**. While this makes them easy to compute, it also means that group fairness metrics cannot tell you *why* there exists a disparity between groups. Disparities can arise in various ways ([Biases as Sources of Unfairness](biases_problem)) an depending on your philosophical framework, each type of bias can necessitate a different intervention ([Choosing the "Right" Fairness Metric](normative_underpinnings)).

## The Impossibility Theorem
At first glance, each of the fairness criteria discussed above seems desirable (*we will dive deeper into this assumption in [Choosing the "Right" Fairness Metric](normative_underpinnings)*). As such, we might want to attempt to satisfy all criteria. Unfortunately, in many cases it is impossible for demographic parity, equalized odds, and equal calibration to hold simultaneously {footcite:p}`kleinberg2016inherent`{footcite:p}`chouldechova2017fair`. 

In particular, when a classifer is not perfectly accurate, it is often impossible to satisfy two of the three criteria at the same time. More specifically:

* If $A$ and $Y$ are not independent, demographic parity and equal calibration cannot hold simultaneously.
* If both $A$ and $R$ are not independent of $Y$, demographic parity and equalized odds cannot hold simultaneously.
* If $A$ and $Y$ are not independent, equalized odds and equal calibration cannot hold simultaneously.

In practice, these conditions will often hold. First of all, in real-world scenarios, classifiers are almost never perfect. 
Second, in cases where fairness is relevant, it is not uncommon that {term}`base rate`s $p_a = P(Y=1 \mid A=a)$ (i.e., the proportion of positives) are different across groups, meaning that $A$ and $Y$ are not independent. 
Finally, for a predictor to be useful, the scores $R$ cannot be independent of $Y$.

### Demographic Parity and Equal Calibration
If base rates differ across groups (i.e., $p_a \neq p_b$), it is impossible for demographic parity and equal calibration to hold at the same time.

This statement is easy to prove when we consider probablistic interpretations of the fairness criteria {footcite:p}`fairmlbook`. 

```{admonition} Proof
A probabilistic interpretation of {term}`demographic parity` requires that the model's predictions are independent of sensitive group membership, i.e., $A \perp\!\!\!\perp \hat{Y}$. {term}`Equal calibration`, on the other hand, requires that, conditional on the model's predictions, the ground truth labels are independent of sensitive group membership, i.e., $A \perp\!\!\!\perp Y \mid \hat{Y}$. 

By the contraction rule for conditional independence, we have:

$$A \perp\!\!\!\perp \hat{Y} \quad and \quad Y \perp\!\!\!\perp A \mid \hat{Y} \quad \Rightarrow \quad A \perp\!\!\!\perp (Y,\hat{Y}) \quad \Rightarrow \quad A \perp\!\!\!\perp Y$$

However, when base rates are unequal, $A$ and $Y$ are **not** independent. As such, demographic parity and equal calibration *cannot* hold when base rates are unequal across groups. Taking the contrapositive complets the proof.
```

Intuitively, enforcing demographic parity when $p_a < p_b$ implies selecting either more positives in group $a$ or more negatives in group $b$ than suggested by the observed 'ground-truth' labels $Y$. If we want to increase the selection rate in group $a$, we must force the classifier to predict more positives than implied by base rate $p_a$. This decreases the group-specific {term}`positive predictive value` (and hence calibration). Similarly, if we were to decrease the selection rate of group $b$ by increasing the number of predicted negatives in group $b$, the {term}`negative predictive value` of group $b$ will deteriorate. In neither of these extremes - as well as cases in between - it is possible to satisfy demographic parity and equal calibration simultaneously.

### Demographic Parity and Equalized Odds
If base rates differ across groups (i.e., $p_a \neq p_b$) and risk scores $R$ are not independent of target variable $Y$, it is impossible for demographic parity and equalized odds to hold at the same time. Lets again consider the probabilistic interpretations of fairness criteria {footcite:p}`fairmlbook`.
.

```{admonition} Proof
{term}`Equalized odds` requires that, conditional on the ground truth labels, the model's predictions are independent of sensitive group membership, i.e., $\hat{Y} \perp\!\!\!\perp A \mid Y $.

In the binary classification scenario, the law of total probability implies that the the following must hold:

$$P(R=r \mid A=a) = P(R=r \mid A=a, Y=1) p_a + P (R=r \mid A=a, Y=0) (1 - p_a)$$

Assuming demographic parity holds (i.e., $A \perp\!\!\!\perp R$) holds, we have:

$$P(R=r) = P(R=r, Y=1) p_a + P (R=r, Y=0) (1 - p_a)$$

Assuming that equalized odds holds (i.e., $A \perp\!\!\!\perp R \mid Y$), gives:

$$P(R=r) = P(R=r \mid Y=1) p + P(R=r \mid Y=0) (1 - p)$$

Let $r_y$ denote $P(R=r \mid Y=y)$. From the previous two findings, if demograhpic parity and equalized odds both hold, we must have:

$$r_1 p_a + r_0(1 - p_a) = r_1 p + r_0 (1-p) \Leftrightarrow p_a (r_1 - r_0) = p(r_1 - r_0) $$ 

which can only hold if either $p_a = p$ (which implies base rates are equal) or $r_1 = r_0$ (which implies that $R$ is independent of $Y$).
```

Again, an intuitive explanation of this result can be found in observing that, when $p_a  < p_b$, demographic parity requires classifying instances differently than suggested by the target variable $Y$. Provided that the (non-thresholded) risk scores $R$ are not independent of the ground truth labels $Y$ (i.e., the model's predicted scores are somewhat sensible), increasing the decision threshold for group $a$ inevitably increases the number of false positives in group $a$ and hence the false positive rate. As before, this also holds for the opposite scenario where we increase the number of negatives in group $b$.

### Equal Calibration and Equalized Odds
When a classifier does not have perfect {term}`accuracy` and {term}`base rate`s differ across sensitive groups, it is impossible to satisfy equal calibration and equalized odds simultaneously. Again, we can prove this using the probabilistic interpretations of the fairness criteria{footcite:p}`fairmlbook`.

```{admonition} Proof
Assuming equalized odds and equal calibration hold, conditional independence gives us:

$$A \perp\!\!\!\perp \hat{Y} \mid {Y} \quad and \quad A \perp\!\!\!\perp Y \mid \hat{Y} \Rightarrow A  \perp\!\!\!\perp (\hat{Y}, Y)$$

The following also holds: 

$$A  \perp\!\!\!\perp (\hat{Y}, Y) \Rightarrow A  \perp\!\!\!\perp \hat{Y} \quad and \quad A \perp\!\!\!\perp Y$$

However, when base rates are unequal, $A$ is not independent of $Y$. As such, equalized odds and equal calibration cannot hold simultaneously when $p_a \neq p_b$. Taking the contrapositive completes the proof.
```

In the binary classification scenario, an alternative proof of this phenomenon follows from the [precision-recall trade-off](precision_recall_curve) {footcite:p}`chouldechova2017fair`.

```{admonition} Alternative proof
Formally, let $\text{tp}$, $\text{tn}$, $\text{fp}$, $\text{fn}$ refer to the number of true positives, true negatives, false positives, and false negatives respectively. Moreover, let $N$ denote the total number of instances (i.e., $N = \text{tp} + \text{tn} + \text{fp} + \text{fn}$). Additionally, let $p = \frac{\text{tp} + \text{fn}}{\text{fp} + \text{tn} + \text{fn} + \text{tp}}$ denote the prevalence of the positive class (i.e., the base rate).

Note the following relationship between the number of false positives $\text{fp}$, the false positive rate ${FPR}$ and the base rate $p$:

$$\text{fp} = \frac{\text{fp}}{\text{fp}+\text{tn}} (\text{fp}+\text{tn}) = FPR (\text{fp} + \text{tn}) = FPR (1 - p) N $$

Similarly, we have:

$$\text{tp} = \frac{\text{tp}}{\text{tp}+\text{fn}} {\text{tp}+\text{fn}} = TPR (\text{tp}+\text{fn}) = (1 - FNR) p \cdot N$$

If we plug in these observations, it is easy to see that the $PPV$ depends on the $FNR$, $FPR$, *as well as* base rate $p$:

$$PPV = \frac{\text{tp}}{\text{tp}+\text{fp}} = \frac{(1-FNR)p}{(1-FNR)p + FPR (1 - p)}$$

Given this observation, if base rates differ across groups (i.e., $p_a \neq p_b$) an imperfect classifier (i.e., $FPR > 0$ and/or $FNR > 0$) that satisfies equal calibration (i.e.,  $PPV_a = PPV_b$) cannot have **both** $FPR_a = FPR_b$ and $FNR_a = FNR_b$.
```

### Fairness: Mission Impossible?
These results seem concerning. However, before we conclude that fairness is impossible to achieve, lets first take a step back. 

From a mathematical point of view, it is impossible to achieve more than two of the discussed fairness metrics at the same time. However, fairness is not merely a mathematical concept. Instead, what these impossibility results reveal is that each of the metrics corresponds to a different - conflicting - view on what it means for a machine learning model to be "fair". That is, although the impossiblity results can be presented as a mathematical dispute, it is better framed as a *moral* dispute. We will revisit this in more detail in the next section, [Choosing the "Right" Fairness Metric"](normative_underpinnings).

## References
```{footbibliography}
```