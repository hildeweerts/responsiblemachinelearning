(fairml)=

# Fairness-Aware Machine Learning

```{warning}
This chapter is still under construction.
```

Fairness-aware machine learning algorithms are algorithmic interventions aimed at mitigating unfairness. A typical approach is to formulate unfairness mitigation as an optimization task, with the goal to achieve high predictive performance whilst satisfying a fairness constraint.

## No Fairness Through Unawareness

Now, you may wonder: if we want to avoid fairness-related harm, why don't we just remove the sensitive feature, i.e., the feature that represents a sensitive characteristic in our data set? Unfortunately, it is not that simple.

While removing a sensitive feature ensures the model cannot explicitly use a sensitive feature as input to a prediction, machine learning models will often still be able to replicate the association between a sensitive feature and the target variable. In its simplest form, one of the remaining features in the data set may act as a proxyvariable for a sensitive feature. The use of proxy variables is common in cases of (subconscious and conscious) discrimination by human actors. A classical example is redlining, a practice in the United States where people were systematically denied services based on their postal code. Neighborhoods that were deemed "too risky" were outlined on the map in the color red. Although postal code may appear to be a neutral feature, it was highly correlated with race. As services were mostly denied in predominantly African-American neighborhoods, African-Americans were indirectly discriminated against. Another example can be found in loan applications. Imagine we want to avoid allocation harm across genders. We decide to exclude the feature that represents gender from our data set, to avoid any direct discrimination. However, if we do include occupation, an attribute which is highly gendered in many societies, the model can still identify historical patterns of gender bias. Here, occupation acts as a proxyvariable for gender: occupation is highly associated with gender.

This is not just a hypothetical problem. **Machine learning algorithms are specifically designed to identify relationships between features in a data set. If undesirable patterns exist within the data, it is very likely that a machine learning model will replicate it.** Removing all possible proxy variables is usually not a viable approach. First of all, it is not always possible to anticipate the patterns through which the sensitive feature can be approximated by the model. Several features that are slightly predictive of the sensitive feature might, taken together, be an accurate predictor of the sensitive feature. Second, apart from their relation with the sensitive feature, proxy variables may provide information that is predictive of the target feature. Removing all features that are slightly related to the sensitive feature could therefore substantially reduce the predictive performance of the model.

Clearly, removing sensitive feature is unlikely to prevent allocation harm, which can still occur in the form of indirect effects. Similarly, patterns of stereotyping and denigration can be deeply embedded in (unstructured) data such as text. Quality-of-service harm, representation harm, and denigration harm can be caused by a lack of informativeness of the data that is available for these groups, which is not solved by removing the sensitive feature either. **To conclude, removing sensitive features is only helpful in achieving a very narrow definition of fairness.** The practical consequence is that it is unlikely that this approach will prevent real-world harm.

## Pre-Processing

## Constrained Learning

## Post-Processing

```{admonition} Summary
:class: tip

...


```

## References

```{footbibliography}

```
