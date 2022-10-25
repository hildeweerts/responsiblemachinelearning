(intro_fairness)=
# Algorithmic Fairness

In recent years, there has been an increasing awareness amongst both the public and scientific community that algorithmic systems can reproduce, amplify, or even introduce unfairness in our societies. From automated resume screening tools that favor men over women to facial recognition systems that fail disproportionately for darker-skinned women. In this chapter, we provide an introduction to algorithmic fairness, list several types of fairness-related harm, and explain why there is no fairness through unawareness.

## Unfair Machines
Machine learning applications often make predictions about people. For example, algorithmic systems may be used to decide whether a resume makes it through the first selection round, judge the severity of a medical condition, or determine whether somebody will receive a loan. Since these systems are usually trained on massive amounts of data, they have the potential to be more consistent than human decision-makers with varying levels of experience. For example, consider a resume screening process. In a non-automated scenario, the likelihood to get through the resume selection round can depend on the personal beliefs of the recruiter who happens to judge your resume. On the other hand, the predictions of an algorithmic resume screening system can be learned from the collective judgement of many different recruiters.

However, the workings of a machine learning model heavily depend on how the machine learning task is formulated and which data is used to train the model. Consequently, prejudices against particular groups can seep into the model in each step of the development process. For example, if in the past a company has hired more men than women, this will be reflected in the training data. The machine learning model is likely to pick up this pattern. 

```{admonition} *Example:* Amazon's Resume Screening Model
In 2015, [Amazon tried to train a resume screening model](https://www.reuters.com/article/us-amazon-com-jobs-automation-insight-idUSKCN1MK08G). In order to avoid biased predictions, the model did not explicitly take the applicant's gender into account. However, it turned out that the model penalized resumes that included terms that suggested that the applicant was a woman. For example, resumes that included the word "women's" (e.g., in "women’s chess club captain") were less likely to be selected. Although Amazon stated the tool “was never used by Amazon recruiters to evaluate candidates”, this incident serves as an example of how machine learning models can, unintentially, replicate undesirable relationships between a sensitive characteristic and a target variable.
```

Notably, the characteristics that could potentially make algorithmic systems desirable over human-decision making, also amplify fairness related risks. One prejudiced recruiter can judge a few dozen resumes each day, but an algorithmic system can process thousands of resumes in the blink of an eye. If an algorithmic system is biased in any way, harmful consequences will be structural and can occur at an exceptionally large scale. Even in applications where predictions do not directly consider individuals, people can be unfairly impacted {footcite:p}`Barocas2019`. For example, a machine learning model that predicts the future value of houses can influence the actual sale prices. If some neighborhoods receive much lower house price predictions than others, this may disproportionately affect some groups over others.

Discrimination and bias of algorithmic systems is not a new problem. Well over two decades ago, {footcite:t}`Friedman1996` analyzed fairness of computer systems. However, with the increasing use of algorithmic systems, it has become clear that the issue is far from solved. Researchers from a range of disciplines have started working on unraveling the mechanisms in which algorithmic systems can undermine fairness and how these risks can be mitigated. 

This has given rise to the research field of {term}`algorithmic fairness`: the idea that algorithmic systems should behave or treat people fairly, i.e., without discrimination on the grounds of {term}`sensitive characteristic`s such as age, sex, disability, ethnic or racial origin, religion or belief, or sexual orientation. Here, sensitive characteristic refers to a characteristic of an individual such that any decisions based on this characteristic are considered undesirable from an ethical or legal point of view. 

Note that this definition of algorithmic fairness is very broad. This is intentional. The concept is applicable to all types of algorithmic systems, including different flavors of artificial intelligence (e.g., symbolic approaches, expert systems, and machine learning), but also simple rule-based systems.

## Types of Harm

The exact meaning of "behaving or treating people fairly" depends heavily on the context of the algorithmic system. There are several different ways in which algorithmic systems can disregard fairness. In particular, we can distinguish between the following types of fairness-related harms{footcite:p}`Madaio2020`:

- {term}`Allocation harm` can be defined as an unfair allocation of opportunities, resources, or information. In our resume selection example, allocation harm can occur when some groups are selected less often than others, e.g., the algorithm selects men more often than women.
- {term}`Quality-of-service harm` occurs when a system disproportionately fails for certain (groups of) people. For example, a facial recognition system may misclassify black women at a higher rate than white men {footcite:p}`buolamwini2018` and a speech recognition system may not work well for users whose disability impacts their clarity of speech {footcite:p}`Guo2019`.
- {term}`Stereotyping harm` occurs when a system reinforces undesirable and unfair societal stereotypes. Stereotyping harm are particularly prevalent in systems that rely on unstructed data, such as natural language processing and computer vision systems. The reason for this is that societal stereotypes are often deeply embedded in text corpora and image labels. For example, an image search for "CEO" may primarily show photos of white men.
- {term}`Denigration harm` refers to situations in which algorithmic systems are actively derogatory or offensive. For example, an automated tagging system may [misclassify people as animals](https://www.theverge.com/2015/7/1/8880363/google-apologizes-photos-app-tags-two-black-people-gorillas) and a chat bot might [start using derogatory slurs](https://fortune.com/2020/09/29/artificial-intelligence-openai-gpt3-toxic/).
- {term}`Representation harm` occurs when the development and usage of algorithmic systems over- or under-represents certain groups of people For example, some racial groups may be overly scrutinized during welfare fraud investigations or neighborhoods with a high elderly population may be ignored because data on disturbances in the public space (such as [potholes](https://hbr.org/2013/04/the-hidden-biases-in-big-data) is collected using a smartphone app. Representation harm can be connected to allocation harms and quality-of-service harms. However, a lack of diversity by itself can already be considered a violation of fairness. Moreover, representation harm can already occur even before the algorithmic system makes a single prediction, which makes it important to consider from the start.
- {term}`Procedural harm` occurs when decisions are made in a way that violates social norms (see e.g., {footcite:t}`Rudin2018a`). For example, penalizing a job applicant for having more experience can violate social norms. Procedural harm is not limited to the prediction-generating mechanisms of the model itself, but can also be extended to the development and usage of the system. For example, is it communicated clearly that an algorithmic decision is made? Do data subjects receive a meaningful justification? Is it possible to appeal a decision?

Note that these types of harm are not mutually exclusive and that this list is not complete -- there may be other context and application specific harms.

## No Fairness through Unawareness
If we do not want to discriminate against certain groups, why don't we just remove the {term}`sensitive feature`, i.e., the feature that represents a sensitive characteristic in our data set? Unfortunately, it is not that simple. To see why, it can help to distinguish between direct discrimination and indirect discrimination.

In European Union law, {term}`direct discrimination` refers to cases where (groups of) individuals are treated less favorably based directly on their membership of a protected-by-law group. In United States laber law, a similar concept is referred to as {term}`disparate treatment`. On the other hand, {term}`indirect discrimination` (similar to {term}`disparate impact` in United States labor law), refers to cases where groups or individuals are treated less favorably based on rules that **seem** neutral, but, as a side effect, disadvantage a protected group.

```{caution}
Much of the terminology in algorithmic fairness research, especially earlier works, are loosely based on legal definitions such as disparate impact and discrimination. However, the legal doctrines to which these terms refer are typically much more complex than the technical metrics suggest. This can result in suggestive terminology that may be misleading for computer scientists who are unaware of the legal complexities and can lead to misinterpretations by legal experts. **In this book, I try to avoid legal terminology unless I refer to the actual legal definition**.
```

An example of direct discrimination is when a person is denied service in a restaurant based on their race. In the context of an algorithmic system, direct discrimination could occur when a machine learning model explicitly uses a sensitive feature to make a prediction. Following this definition of fairness, removing the sensitive feature will prevent discrimination. A classic example of *indirect* discrimination is redlining. This refers to a practice in the United States where people were systematically denied services based on their postal code. Neighborhoods that were deemed "too risky" were outlined on the map in the color red. Although postal code may appear to be a neutral feature, it was highly correlated with race. As services were mostly denied in predominantly African-American neighborhoods, African-Americans were indirectly discriminated against. Another example can be found in loan applications. Imagine we want to avoid allocation harm across genders. We decide to exclude the feature that represents gender from our data set, to avoid any direct discrimination. However, if we do include occupation, an attribute which is highly gendered in many societies, the model can still identify historical patterns of gender bias. Here, occupation acts as a {term}`proxyvariable` for gender: occupation is highly associated with gender. In the case of indirect discrimination, variables included in our model may unintentionally act as a proxy variables for a sensitive characteristic. 

This is not just a hypothetical problem. **Machine learning algorithms are specifically designed to identify relationships between features in a data set. If undesirable patterns exist within the data, it is very likely that a machine learning model will replicate it.** Removing all possible proxy variables is usually not a viable approach. First of all, it is not always possible to anticipate the patterns through which the sensitive feature can be approximated by the model. Several features that are slightly predictive of the sensitive feature might, taken together, be an accurate predictor of the sensitive feature. Second, apart from their relation with the sensitive feature, proxy variables may provide information that is predictive of the target feature. Removing all features that are slightly related to the sensitive feature could therefore substantially reduce the predictive performance of the model.

Clearly, removing sensitive feature is unlikely to prevent allocation harm, which can still occur in the form of indirect effects. Similarly, patterns of stereotyping and denigration can be deeply embedded in (unstructured) data such as text. Quality-of-service harm, representation harm, and denigration harm can be caused by a lack of informativeness of the data that is available for these groups, which is not solved by removing the sensitive feature either. **To conclude, removing sensitive features is only helpful in achieving a very narrow definition of fairness.** The practical consequence is that it is unlikely that this approach will prevent real-world harm.

```{admonition} Summary
:class: tip

**Unfair Machines**

Machine learning models can reproduce, amplify, and even introduce unfairness. This has given rise to the research field of algorithmic fairness: the idea that algorithmic systems should behave or treat people without discrimination on the grounds of sensitive characteristics.

**Types of harm**

There are different ways in which algorithmic systems can disregard fairenss, including {term}`allocation harm`, {term}`quality-of-service harm`, {term}`stereotyping harm`, {term}`denigration harm`, {term}`representation harm`, and {term}`procedural harm`.

**No fairness through unawareness**

As machine learning models are designed to identify relationships between features in the data set, removing sensitive features is generally not sufficient to mitigate the risk of fairness-related harm.

```

## References

```{footbibliography}
```