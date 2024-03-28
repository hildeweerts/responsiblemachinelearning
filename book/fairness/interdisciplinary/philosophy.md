(philosophy_egalitarianism)=

# Philosophy: What is "Fair"?

What does it mean for a machine learning model to be "fair", "non-discriminatory", or "just"? Questions of what is right or wrong have long been studied by philosophers studying ethics. Formally, ethics is a branch of philosophy that considers a systematic reflection on _morality_: the totality of opinions, decisions, and actions with which people express, individually or collectively, what they think is good or right {footcite:p}`poel2011`.

We can distinguish two major branches of ethics. _Descriptive_ ethics considers describing existing morality. This branch of ethics involves describing and analyzing how people live to draw general conclusions about what they consider moral. _Normative_ ethics, on the other hand, is a branch of ethics that involves formulating how to judge what is right and what is wrong. When we talk about ethics in the context of machine learning, we usually consider normative decision-making.

At the basis of normative decision-making lie values. Values are beliefs about what is important ('valuable') in life. In the context of normative ethics, moral values refer to a person's general beliefs about what is right and what is wrong. {footcite:p}`poel2011` define moral values as follows: convictions or matters that people feel should be strived for in general and not just for themselves to be able to lead a good life or to realize a just society. Examples of values are honesty, compassion, fairness, courage, and generosity. Within a society, shared values may be translated into a set of rules about how people ought to act. Such a rule, which prescribes what actions are required, permitted, or forbidden is often referred to as a moral _norm_.

Considering concepts of fairness, justice, and discrimination, perhaps one of the most influential schools of thought in philosophy is _egalitarianism_. In this section, we consider algorithmic fairness metrics and mitigation algorithms through this lens.

(egalitarianism)=

## Egalitarianism

Group fairness metrics require some form of equality across groups. Given this characteristic, several scholars have suggested that the philosophical perspective of egalitarianism may provide an ethical framework to understand and justify group fairness metrics {footcite:p}`binns2018fairness`{footcite:p}`heidari2019moral`{footcite:p}`hertweck2021moral`{footcite:p}`weerts2022does`. Egalitarianism is centered around distributive justice: the just allocation of benefits and burdens.

Egalitarian theories are generally grounded in the idea that all people are equal and should be treated accordingly. An important concept within egalitarianism is _equality of opportunity_: the idea that (1) social positions should be open to all applicants who possess the relevant attributes, and (2) all applicants must be assessed only on relevant attributes {footcite:p}`arneson2018four`. There are two main interpretations of equality of opportunity.

```{note}
While the [equal opportunity](equal_opportunity) fairness metric was inspired by a notion of equality of opportunity, the metric does not capture more nuanced philosophical notions of equality of opportunity.
```

A _formal_ interpretation of equality of opportunity requires all people to formally get the opportunity to apply for a position. Applicants are to be assessed on their merits, i.e., according to appropriate criteria. In particular, (direct) discrimination based on arbitrary factors, particularly sensitive characteristics such as race or gender, is prohibited. Note that formal equality of opportunity does not require applicants of all sensitive groups to have a non-zero probability to be selected. In particular, formal equality of opportunity allows the use of criteria that are (highly) related to sensitive characteristics, provided these criteria are relevant for assessing merit.

Substantive theories go further and pose that everyone should also get a substantive opportunity to _become_ qualified. In particular, John Rawls' theory of justice {footcite:p}`rawls1971theory` requires everyone with similar innate talent and ambition to have similar prospects for success, irrespective of their socio-economic background.

## Fairness Metrics as Egalitarian Goals

A central debate in egalitarian theories of justice is _what_ should be equal {footcite:p}`sen1979equality`. When we consider fairness metrics through an egalitarian lens, we can see that each metric provides a different answer to this question.

For example, measuring fairness as [demographic parity](demographic_parity) implies that each group of individuals is, on average, equally deserving of $\widehat{Y}$, regardless of their ground-truth class $Y$. For example, in recruitment, we may require that men and women are hired at equal rates. In clinical decision-making, we may require that different racial groups receive healthcare at equal rates. [Equalized odds](equalized_odds), on the other hand, assumes that each group of individuals with the same ground-truth class $Y$ are, on average, equally deserving of $\widehat{Y}$. For instance, we may require that _qualified_ men and women are hired at equal rates (and vice versa) or that _sick_ patients of different ethnicities receive healthcare at equal rates. [Equal calibration](equal_calibration), finally, requires that predicted scores have similar meaning, irrespective of sensitive group membership. For example, equal calibration requires that a female patient who receives a mortality risk score of 0.4 has a similar probability of mortality as a male patient that receives a score of 0.4.

A direct consequence of these different valuations of predicted outcomes is that under many circumstances, demographic parity, equalized odds, and equal calibration are [mathematically incompatible](impossibility_theorem) {footcite:p}`kleinberg2016inherent`.

So how do we choose one over the other? For this, we need to unravel the normative and empirical assumptions that underly each of these fairness metrics.

### Demographic Parity

Demographic parity does not take into account the true label $Y$. Consequently, if $P(Y=1|A=a) \neq P(Y=1)$, demographic parity rules out a perfect predictor. In other words, if base rates are different across groups, satisfying demographic parity requires one to make predictions that do not coincide with the observed outcomes. For this reason, {footcite:p}`wachter2020bias` refers to demographic parity as a bias-_transforming_ metric: it requires us to change the status quo.

Some scholars have argued that demographic parity may be justified as a measure of fairness if the data that was used to train the model is affected by [historical bias](historical_bias): social biases are explicitly encoded in the data in the form of an association between a sensitive feature and the target variable {footcite:p}`hertweck2021moral`.

```{figure} ../../figures/historicalbias.svg
:name: historicalbias
:width: 600px
:align: center
:alt: A graphical representation of different types of historical bias. The picture contains three primary objects; 'potential - one's inate potential to become a data scientist', 'construct - employee quality', and 'measurement - hiring decisions'. An arrow is drawn between 'potential' and 'construct', which is annotated with '(un)just life bias, innate potential is not equal to employee quality'. A second arrow is drawn between 'construct' and 'measurement', which is annotated with 'measurement bias, employee quality is not equal to hiring decisions'.
:target: none

```

Under _measurement bias_, the data paints an incorrect or incomplete picture of the decision subject's true qualities. For example, if a patient's healthcare needs are measured as healthcare costs, the healthcare needs of patients who have poor healthcare insurance will be underestimated. From a moral perspective, it seems that decision subjects should not have to bear the harmful consequences of the incorrect beliefs decision-makers have about them. A decision-making policy that relies on such incorrect beliefs can be understood as a violation of formal equality: decision subjects are not assessed based on appropriate criteria but on irrelevant characteristics (insurance policy). Under the empirical assumption that the true distribution of positives and negatives is equal between groups, a fair distribution of positives and negatives would satisfy demographic parity.

Under _life's bias_, the data may accurately represent reality, but reality represents an unequal status quo. For example, we may assume empirically that on average, women have the same innate potential as men to become good data scientists. However, societal gender stereotypes may have affected study progress and career choices, resulting in an unequal distribution of qualified candidates across genders. From a moral perspective, we may assume that women should not have worse job prospects compared to men. This would clearly violate substantive equality: applicants with a similar innate potential do not have equal prospects. Again, under certain assumptions, a fair distribution of positives and negatives would satisfy demographic parity.

### Equalized Odds

Two distinct problems can lie at the root of unequal odds in machine learning. First, a model may be less able to accurately distinguish between positives and negatives for some group compared to another group (see {numref}`nonoverlappingcurves`). In this case, equalizing odds requires improving the predictive abiltiy of the model for the worst-off group or purposefully reducing the preditice ability for the best-off group.

```{figure} ../../figures/nonoverlappingcurves.svg
:name: nonoverlappingcurves
:width: 200px
:align: center
:target: none

Two group-specific ROC curves which do not overlap. The model is less able to distinguish between positives and negatives for Group $A$ (yellow) compared to Group $B$ (purple). Apart from trivial endpoints, equalizing the odds requires either improving the predictive ability of the model for Group $A$ or reducing predictive ability for Group $B$.

```

When base rates are equal between sensitive groups, a requirement of equal odds corresponds to a requirement of equal predictive ability. The underlying moral assumption of objecting unequal odds in this scenario is similar as measurement bias: decision subjects should not have to bear the harmful consequences of incorrect beliefs (i.e., predictions) decision-makers have about them. Consequently, one could argue that a fair distribution of false positives and false negatives satisfies equalized odds.

A second cause of unequal odds is disparities in base rates, which can result in a different trade-off between false positives and false negatives between sensitive groups (see {numref}`nonoverlappingcurves`). In this case, equalizing odds corresponds to setting group-specific decision thresholds.

```{figure} ../../figures/disparatebaserates.svg
:name: disparatebaserates
:width: 200px
:align: center
:target: none

Two group-specific ROC curves that overlap exactly. If Group $A$ (yellow) and Group $B$ (purple) have different base rates, the same decision threshold ends up on a different spot on the group-specific ROC curve. Equalizing the odds corresponds to setting group-specific decision thresholds.
```

If risk scores are well-calibrated, group-specific decision thresholds correspond to differences in group-specific misclassification cost: a false positive for one Group $A$ is considered less costly than a false positive for Group $B$. Potential moral justifications relate to the cause of the disparity in base rates, i.e., measurement bias or (unjust) life's bias. For example, one could argue that assigning different misclassification costs to a particular sensitive group is justified if, in a certain society, it is already disproportionately costly to belong to a marginalized group.

### Equal Calibration

Equal calibration quantifies an understanding of fairness that a score should have the same _meaning_, regardless of sensitive group membership. Similar to equalized odds, the underlying assumption is that the target variable is a reasonable representation of what reality looks or should look like. However, as opposed to equalized odds, equal calibration does not acknowledge that the relationship between features and the target variable may be different across groups.

As opposed to demographic parity and equalized odds, requiring equal calibration usually does not require an active intervention. That is, we often get equal calibration "for free" when we use machine learning approaches. As such, learning without explicit fairness constraints often implicitly optimizes for equal calibration {footcite:p}`liu2019implicit`.

## Enforcing Fairness Constraints

In some cases, a fair distribution of prediction corresponds to a specific fairness constraint. However, simply enforcing a fairness constraint through a fairness-aware machine learning technique does not always lead to a justifiable distribution of burdens and benefits and can have undesirable side effects.

First of all, acknowledging that the target variable may be affected by measurement bias does not imply that _all_ solutions that satisfy demographic parity are beneficial.

````{admonition} Example: Disease Detection under Measurement Bias

Imagine we have a classifier that is the best way to diagnose a disease that predicts a positive if and only if receiving an invasive but preventative treatment is beneficial to the patient. Two sensitive groups, Group $A$ and Group $B$ can be distinguished in the patient population.

Both patient populations are equally likely to be sick, with a true base rate of 0.5. Unfortunately, as it turns out, data collection is affected by measurement bias: individuals in group $B$ are more often asymptomatic, making it harder to detect the disease in these patients compared to patients who belong to group $A$. As a result, the number of sick patients in group $B$ are undercounted, resulting in a lower base rate (i.e., proportion of positives).

```{figure} ../../figures/diseasedetection.svg
:name: diseasedetection
:width: 250px
:align: center
:target: none

Even if we acknowledge that the collected data is affected by measurement bias, we don’t know which instances in group $B$ would benefit from the treatement.

```

As patients are equally likely to get sick, it seems a fair classifier would diagnose patients of group $A$ and group $B$ at equal rates, satisfying demographic parity. However, _enforcing_ demographic parity algorithmically requires us to diagnose more patients in group $B$ than suggested by the data. However, by stipulation, our classifier is the best way to diagnose patients. As such, without further information, we have no idea exactly which patients would benefit from the treatment.

Randomly selecting more patients from group $B$ in order to satisfy demographic parity is very likely to subject healthy patients to invasive treatment that does not benefit them.

Put differently: even if a similar proportion of individuals would benefit from treatment in group $A$ and group $B$, this does not imply we should enforce similar treatment at all costs.

````

The above example illustrates that even if the fair distribution of positives and negatives satisfies demographic parity, _enforcing_ equal selection rates at all costs may not result in said fair distribution.

Simply enforcing demographic parity may also be problematic when a target variable represents an unequal status quo. For example, it seems that a Rawlsian take on employment would require that men and women have equal job prospects. In particular, a situation in which women's job prospects are affected by societal gender stereotypes, leading to an unequal distribution of qualifications across genders, does not satisfy substantive equality. One could naively interpret this as an argument to enforce demographic parity in a resume selection algorithm. However, is it beneficial for less qualified women to get a job they cannot adequately perform? At an individual level, such a policy could lead to low job satisfaction and stress. At a societal level, hiring less qualified women could reinforce existing gender stereotypes.

Instead, true substantive equality requires providing everybody the opportunity to _become_ qualified for a social position. This brings us to a thorny issue: it is often difficult if not impossible to meaningfully address social inequalities at the level of a single decision-making point. Instead, substantive equality would require such decisions to be accompanied by a broader set of additional interventions that ensure a hired candidate can be successful in their job, such as vocational training programs, additional resources for on-the-job learning, or mentoring programs.

Naively enforcing group fairness constraints can also have other undesirable side effects. Most objections to egalitarianism revolve around the central egalitarian view that the presence or absence of inequality is what matters for justice. Alternative principles of distributive justice include, for example, _maximin_, which requires the expected welfare of the worst-off group to be maximized. In particular, anti-egalitarianist philosophers often invoke the _leveling down objection_ {footcite:p}`parfit1995equality`, which points out that equality can be achieved through lowering the welfare of the better-off to the level of the worse-off, without making those worse-off any better-off in absolute terms. In machine learning, fairness constraints are often implemented as equality constraints, raising similar concerns. A concrete example is [randomized group-specific decision thresholds](randomized_thresholds). Specifically, if group-specific ROC curves do not intersect, the randomization of group-specific decision thresholds achieves equalized odds by _decreasing_ the performance for the better-off group ({numref}`levelingdown`).

```{figure} ../../figures/randomizedthresholdsroc.svg
:name: levelingdown
:width: 300px
:align: center
:target: none

In this case, the group-specific ROC curves do not intersect, so we cannot find non-randomized group-specific decision thresholds such that equalized odds is satisfied. Equalized odds is achieved because by realizing suboptimal performance for the better-off group (Group 1).
```

These examples show that fairness constraints are very simplistic measurements of more nuanced notions of fairness. The complexity of fairness-aware machine learning algorithms can disguise the underspecification of fairness constraints. Simple algorithms, such as [relabeling](relabeling) and [reject option classification](reject_option_classification), are easy to interpret as a decision-making policy. For example, consider the reject option classification algorithm. This algorithm assumes the availability of risk scores and reclassifies instances for which the classifier is the most uncertain, i.e., closest to the decision boundary to satisfy demographic parity. Implicitly, the reject option classification approach relies on the empirical assumption that the risk scores accurately capture uncertainty of belonging to the positive class. An accompanying normative assumption is that instances near the decision boundary are more deserving of a positive prediction than instances far away from the decision boundary. Of course, the suitability of reject option classification highly depends on whether these assumptions hold up in practice.

Simple algorithms typically seem very intrusive: we actively adjust the training data or predictions in a specific way. At first glance, more complex algorithms, such as [representation learning](representation_learning) and [adversarial learning](adversarial_learning) appear more sophisticated. However, both simple and complex algorithms suffer from the same underspecification of fairness constraints and do not necessarily lead to better real-world outcomes. More complex fairness-aware machine learning algorithms make it difficult to discern the learned decision-making policy. In turn, this makes it hard to meaningfully discuss the underlying empirical and normative assumptions.

## Concluding Remarks

In this section, we have seen that philosophical theories can provide normative grounding for the choice of a fairness metric. We have also seen that mathematical fairness constraints are simplified notions of fairness. While under some circumstances, a fair classifier will satisfy certain fairness metrics, simply enforcing mathematical fairness constraints using a fair-ml technique may not result in a fair distribution of outcomes and can have several undesirable side effects.

```{note}
Parts of this section were adapted from {footcite:t}`weerts2022does` and {footcite:t}`weerts2024can`.
```

## References

```{footbibliography}

```
