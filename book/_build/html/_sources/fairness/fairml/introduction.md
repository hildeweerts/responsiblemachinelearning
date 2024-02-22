(fairml)=

# Fairness-Aware Machine Learning

So far, we have covered various ways in which machine learning systems can result in unfair outcomes, several biases that lie at the root of fairness-related harm, and a set of metrics that can be used to measure harm. A natural question, then, is: how can we ensure that our machine learning model adheres to a certain fairness constraint? Over the past few years, computer scientists have been productive in developing various algorithms aimed at mitigating unfairness.

## No Fairness Through Unawareness

Now, you may wonder: if we wish to avoid fairness-related harm, why don't we just remove the sensitive feature from our data set? Unfortunately, it is not that simple.

While removing a sensitive feature ensures the model cannot explicitly use a sensitive feature as input to a prediction, machine learning models will often still be able to replicate the association between a sensitive feature and the target variable. In its simplest form, one of the remaining features in the data set may act as a proxy variable for a sensitive feature. The use of proxy variables is common in cases of (subconscious and conscious) discrimination by human actors. A classical example is redlining, a practice in the United States where people were systematically denied services based on their postal code. Neighborhoods that were deemed "too risky" were outlined on the map in the color red. Although postal code may appear to be a neutral feature, it was highly correlated with race. As services were mostly denied in predominantly African-American neighborhoods, African Americans were disproportionately affected by this practice. Another example can be found in loan applications. Imagine we want to avoid allocation harm across genders. We can decide to exclude the feature that represents gender from our data set, to avoid any direct discrimination. However, if we do include occupation, an attribute that is highly gendered in many societies, the model can still identify historical patterns of gender bias. Here, occupation can unintentionally act as a proxy variable for gender.

As an increasing number of empirical evidence highlights, this is not just a hypothetical problem {footcite:p}`kamiran2012data` {footcite:p}`kamishima2012fairness`. Machine learning algorithms are specifically designed to identify relationships between features in a data set. If undesirable patterns exist within the data, a machine learning model will likely replicate it. Removing all possible proxy variables is usually not a viable approach. First of all, it is not always possible to anticipate the patterns through which the sensitive feature can be approximated by the model. Several features that are slightly predictive of the sensitive feature might, taken together, accurately predict a sensitive feature. Second, apart from their relation with the sensitive feature, proxy variables may provide information that is predictive of the target feature. Removing all features that are slightly related to the sensitive feature could therefore substantially reduce the predictive performance of the model.

Removing sensitive features is unlikely to prevent allocation harm, which can still occur in the form of indirect effects. Similarly, patterns of stereotyping and denigration can be deeply embedded in (unstructured) data such as text. Quality-of-service harm, representation harm, and denigration harm can be caused by a lack of informativeness of the data that is available for these groups, which is not solved by removing the sensitive feature either. To conclude, removing sensitive features is only helpful in achieving an extremely narrow notion of fairness. The practical consequence is that it is unlikely that this approach will prevent real-world harm.

## Fairness-aware Machine Learning Algorithms

A common approach in computer science literature on algorithmic fairness is to formulate unfairness mitigation as an optimization task, to achieve high predictive performance whilst satisfying a fairness constraint. Generally speaking, we can distinguish three types of approaches that intervene in different parts of the machine learning process.

- **Pre-processing algorithms.** Pre-processing algorithms make adjustments to the data to mitigate the unfairness of the downstream machine learning model.
  Most pre-processing algorithms are designed to obscure undesirable associations between one or more sensitive features and the target variable of the model.
- **Constrained learning algorithms.** Constrained learning techniques directly incorporate a fairness constraint in the learning algorithm, typically by adjusting existing learning paradigms. For example, a typical constrained learning approach adjusts the loss function of an existing machine learning algorithm to penalize unfair predictions. This category also includes wrapper methods, which enforce fairness constraints via optimization external to the machine learning algorithm itself, such as hyperparameter optimization or ensemble learning.
- **Post-processing algorithms.** Post-processing approaches adjust an existing model, either through post-processing predictions (e.g., by shifting the decision thresholds) or by adjusting the learned model parameters directly (e.g., the coefficients of a linear regression model or the labels assigned to the leaves of a decision tree.)

The dividing line between different approaches is not always clear-cut. For example, data resampling straddles between pre-processing and constrained learning approaches, while decision threshold optimization can be implemented as a metric-agnostic hyperparameter search or a post-processing approach that takes into consideration the nature of the fairness constraint that is to be enforced.

In the remainder of this chapter, we will discuss several simple fairness-aware machine learning algorithms. The goal is not to present a comprehensive overview of all the different techniques that have been proposed in the algorithmic fairness literature. Instead, these examples allow you to develop an intuition of the advantages, disadvantages, and limitations of the fairness-aware machine learning paradigm.

```{admonition} Chapter Summary
:class: tip

[**Pre-processing Algorithms**](fairml_pre_processing)

Pre-processing algorithms make adjustments to the data to mitigate the unfairness of the downstream machine learning model. In this section, we will discuss the following approaches:

* [Relabelling](relabeling)
* [Representation learning](representation_learning)

[**Constrained Learning Algorithms**](fairml_constrained_learning)

Constrained learning techniques directly incorporate a fairness constraint in the learning algorithm, typically by adjusting existing learning paradigms. In this section, we will discuss the following approaches:

* [Reweighing](reweighing)
* [Regularization](regularization)
* [Decoupled Classification](decoupled_classification)
* [Adversarial Learning](adversarial_learning)

[**Post-processing Algorithms**](fairml_post_processing)

Post-processing approaches adjust an existing model, either through post-processing predictions or by adjusting the learned model parameters directly. In this section, we will discuss the following approaches:

* Reject-Option Classification
* Randomized Decision Thresholds
* Leaf Relabelling

```

## References

```{footbibliography}

```
