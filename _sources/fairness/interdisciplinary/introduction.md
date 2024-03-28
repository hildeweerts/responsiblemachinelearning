(intro_interdisciplinary)=

# Interdisciplinary Perspectives on Fair-ML

Computer scientists have been productive in developing fairness-aware machine learning algorithms. But which algorithm is most suitable in which scenario?

From a technical perspective, algorithm selection of fair-ml algorithms does not differ much from the selection of other approaches in a data scientist's toolbox. Most algorithms can easily be integrated into existing [model selection](model_selection) pipelines and evaluated accordingly.

From a practical perspective, various characteristics could be relevant:

- _Machine learning algorithm._ Some fair-ml algorithms are designed specifically for a particular machine learning algorithm, while others are model-agnostic. For example, [adversarial learning](adversarial_learning) and [leaf relabeling](leaf_relabeling) are designed for neural networks and decision trees respectively, while [relabeling](relabeling) and [rejectoptionclassification](reject_option_classification).
- _Fairness constraints._ Is the algorithm designed with one or more particular fairness constraints in mind (e.g., [demographic parity](demographic_parity) or [`equalized odds`](equalized_odds)) or does it also allow custom constraints?
- _Sensitive feature_. Some algorithms are designed specifically for binary sensitive features, while others take into consideration categorical and numerical sensitive features as well as intersectional groups.
- _Access to sensitive features at prediction time._ Does the algorithm require access to sensitive features at prediction time or only during training?
- _Computational complexity._ Is the algorithm computationally cheap or expensive?

However, one crucial question remains: **does the application of a fairness-aware machine learning algorithm _actually_ lead to fairer outcomes?**

In this chapter, we will leverage interdisciplinary insights to help answer this question.

```{admonition} Chapter Summary
:class: tip

[**Philosophy**](philosophy_egalitarianism)

We consider fairness metrics and mitigation algorithms through the lens of egalitarianism, a school of thought in political philosophy. We show that different fairness metrics correspond to different understandings of what ought to be equal between groups. However, even when a fair distribution of outcomes corresponds to a particular fairness constraint, enforcing that constraint algorithmically does not always lead to the fair distribution and can have unintended side effects.

[**Law**](law_eu)

In this section, we explore the requirements set by EU non-discrimination law and how it applies to fairness of machine learning systems.

[**Science and Technology Studies**](abstraction_traps)

In the translation of a real-world problem to a machine learning task, data scientists and researchers may fall into what {footcite:p}`Selbst2019` refer to as an _abstraction trap_. In this section, we discuss several of these traps - and what you can do to avoid them.

```

## References

```{footbibliography}

```
