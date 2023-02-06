(normative_underpinnings)=
# Moral Philosophy: Choosing the "Right" Fairness Metrics

```{warning}
This section is still under construction.
```
<!--

### Types of harm
Several types of [fairness-related harm](types_of_harm), such as {term}`allocation harm` and {term}`quality-of-service harm`, revolve around the extent to which some groups of individuals are worse-off, on average, compared to others.  More specifically, {term}`group fairness metric`s  

If the output of the model corresponds to some kind of resource (e.g., a job or a loan) and selection rates differ across groups, there is a risk of {term}`allocation harm`. For example, in our hiring example, the selection rate of applicants who identify as men may be higher compared to other applicants, i.e., relatively more men are classified as positive compared to women or non-binary applications.

For example, in a hiring scenario, we may mistakingly reject strong female candidates more often than strong male candidates. The risk of quality-of-service harm is particularly prevalent if the relationship between the features and target variable is different across groups. The risk is further amplified if less data is available for some groups.  For example, strong candidates for a data science position may have either a quantitative social science background or a computer science background. Now imagine that in the past, hiring managers have mostly hired people with a computer science degree but hardly any social scientists. As a result, a machine learning model could mistakingly penalize people who do not have a computer science degree. If particular groups are overrepresented in the candidate pool of social scientists, the error rates may be be higher for those groups, resulting in a quality-of-service harm.

In EU and US law, indirect discrimination in employment may not be unlawful if it is justified by a so-called "legitimate aim". Examples of legal justifications for discrimination are genuine occupational requirement and business necessity. For example, a film producer is allowed to hire only male actors to play a male role, as this is considered a genuine occupational requirement.  In particular, EU discrimination law is highly contextual - requiring normative or political evaluations on a case-by-case basis {footcite:p}`wachter2021fairness`.

First, every quantitative fairness metric is necessarily a simplification, lacking aspects of the substantive nature of fairness long debated by philosophers, legal scholars, and sociologists~\shortcite{selbst-fat19,jacobs-facct21,schwobel-facct22a,chen22}. Determining which algorithmic fairness metric is most appropriate for a given use case is still an active area of research~\shortcite{hellman-vlr20,wachter-clsr21,hertweck-facct21,hedden-ppa21}.

### Assessment versus Optimization
Third, while group fairness metrics can help to \emph{assess} potential fairness-related harms, it can be challenging to anticipate all relevant side effects of technical interventions that \emph{enforce} them. In particular, group fairness metrics such as demographic parity and equalized odds are parity-based, meaning they enforce parity of statistics, such as error rates, across sensitive groups. However, group fairness metrics do not set specific constraints on the exact distribution of predictions. For example, it is theoretically possible to achieve equal false positive rates by increasing the false positive rate of the better-off group such that it is equal to that of the worst-off group, which in most cases will not make the worst-off group any better off. The under-specification of fairness metrics is even more problematic when we consider bias-transforming metrics such as demographic parity. Borrowing an example from~\shortciteA{dwork-itcsc12}, it is possible to increase the selection rate for female applicants by inviting the least qualified female applicants for an interview. While this intervention would satisfy demographic parity, it is unlikely that any of the unqualified applicants would actually make it to the next round. Although extreme `solutions, such as this example, are unlikely to end up on the Pareto front of a \fautoml{} solution, more subtle variations can be hard to detect. Without careful modeling of measurement bias and historical bias, simply enforcing demographic parity -- while theoretically an important measure for equitable outcomes -- can have undesirable side effects that can even harm the groups the intervention was designed to protect~\shortcite{weerts-arxiv22}.

#### When should we use demographic parity as a fairness metric?
The [underlying assumption about fairness of demographic parity](https://arxiv.org/abs/1609.07236) is  that, **regardless of what the measured target variable says**, either:
1. *Everybody **is** equal*. For example, we may believe that traits relevant for a job are independent of somebody's gender. However, due to social biases in historical hiring decisions, this may not be represented as such in the data.
2. *Everybody **should be** equal*. For example, we may believe that different genders are not equally suitable for the job, but this is due to factors outside of the individual's control, such as lacking opportunities due to social gender norms.

Enforcing demographic parity might lead to differences in treatment across sensitive groups, causing otherwise similar people to be treated differently. For example, two people with the exact same features, apart from race, would get a different score prediction. This can be seen a form of *procedural harm*. Consequently, demographic parity is only a suitable metric if one of the two underlying assumptions (everybody *is* or *should be* equal) holds. A limitation of demographic parity is that it does not put any constraints on the scores. For example, to fulfill demographic parity, you do not have to select the most risky people from different racial groups as long as you pick the same proportion for each group. 

#### When should we use equalized odds as a fairness metric?
If error rates differ across groups, there is a risk of **quality-of-service harm**.


Equalized odds quantifies the understanding of fairness that we should not make more mistakes for some groups than for other groups. Similar to demographic parity, the equalized odds criterion acknowledges that the relationship between the features and the target may differ across groups and that this should be accounted for. However, as opposed to the *everybody is or should be equal* assumptions of demographic parity, **equalized odds implicitly assumes that the target variable is a good representation of what we are actually interested in**.

#### When should we use equal calibration as a fairness metric?

Equal calibration quantifies an understanding of fairness that a score should have the same *meaning*, regardless of sensitive group membership. Similar to equalized odds, the underlying assumption is that the target variable is a reasonable representation of what reality looks or should look like. However, as opposed to equalized odds, equal calibration does not acknowledge that the relationship between features and target variable may be different across groups.

As opposed to demographic parity and equalized odds, requiring equal calibration usually does not require an active intervention. That is, we usually get equal calibration "for free" when we use machine learning approaches. As such, learning without explicit fairness constraints often [implicitly optimizes for equal calibration](https://arxiv.org/abs/1808.10013).

## Error versus Associations
Note that demographic parity does not take into account the true label $Y$ and, consequently, if $P(Y=1|A=a) \neq P(Y=1)$, demographic parity rules out a perfect predictor. In other words, if base rates are different across groups, satisfying demographic parity requires one to make predictions that do not coincide with the observed outcomes. For this reason, \shortciteA{wachter-vlr2020} refers to demographic parity as a bias-\emph{transforming} metric: it requires us to change the status quo. There are two primary reasons why we would like to do so. First of all, $Y$ may be subject to \emph{measurement bias}~\shortcite{jacobs-facct21}. For example, in predictive policing, rearrests may be an inaccurate proxy for actual recidivism due to biased policing practices that inflate the base rates for some groups. Second, base rates may differ across groups due to an unjust status quo, which is sometimes referred to as \emph{historical bias}~\shortcite{suresh-eaamo21a}. In this case, demographic parity is often supported by some form of an egalitarian argument: when valuable resources such as jobs and loans are allocated to people, these should be distributed equally regardless of characteristics such as gender and race~\shortcite{weerts-arxiv22}.

As opposed to demographic parity, equalized odds does explicitly take into account $Y$. It is therefore what \shortciteA{wachter-vlr2020} refers to as a bias-\textit{preserving} metric: optimizing for equalized odds will preserve the status quo as much as possible, implicitly assuming that any measurement bias or historical bias present in the data should be preserved.

### Disparate Impact / Direct Discrimination

### Equality of Opportunity

# Connection to Non-Discrimination Law
Previous work has often cited the \emph{four-fifths} rule~\shortcite<e.g.,>{feldman-kdd15a} as an example of such a constraint, but this rule only holds in a very narrow domain of US labor law and translating such legal requirements into fairness metrics requires multiple abstractions possibly invalidating the result of a fairness metric~\shortcite{chen22}. Similarly, EU anti-discrimination law is designed to be context-sensitive and reliant on interpretation~\shortcite{wachter-clsr21}, making it challenging to set any hard constraints in advance. We argue that fairness metrics should not be equated with legal fairness principles and that any constraints loosely derived from legal texts should be questioned.

### Socially Salient Groups
What groups should we consider? *How* should we measure these groups?

During the evaluation stage, the final model is scrutinized in more detail. {term}`evaluation bias` refers to the use of performance metrics and procedures that are not appropriate for the way in which the model will be used {footcite:p}`suresh2020`. {footcite:t}`Mitchell2018` identify several underlying assumptions of performance metrics. First, these metrics assume that individual decisions are independent of each other. Note how this assumption is grounded in utilitarianism, in which overall utility is expressed as the sum of individual utilities. In practice, however, the impact of a decision may not be independent across instances. For example, denying one family member a loan may impact another family member's ability to repay their own loan. Additionally, it is typically assumed that decisions are symmetrical, i.e. the impact of the outcome is equal across instances. Again, this often does not hold in practice. For example, a rejection of a job application can have a very different impact depending on whether that person is currently employed or unemployed.


-->

## References
```{footbibliography}
```