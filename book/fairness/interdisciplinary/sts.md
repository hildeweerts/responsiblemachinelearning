(abstraction_traps)=

# Science and Technology Studies: Abstraction Traps

Translating a real-world problem into a machine learning task is not easy. By definition, a model is a simplification of reality. A data scientist's task is to decide which elements of the real world need to be included in the model and which elements will be left out of scope. To this end, some amount of abstraction is required. This involves removing details to focus attention on general patterns. The impact of your system, both positive and negative, highly depends on how you define the machine learning task. By abstracting away the context surrounding a model and its inputs and outputs, you may accidentally abstract away some of the consequences as well -- leaving them unaccounted for. A mismatch between the machine learning task and the real-world context is referred to as an abstraction trap {footcite:p}`Selbst2019`.

## The Framing Trap

Machine learning models hardly ever operate in isolation. A decision-making process may incorporate (multiple) other machine learning models or human decision-makers. The framing trap considers the failure to model the relevant aspects of the larger system your machine learning model is a part of.

For example, consider a scenario in which judges need to decide whether a defendant is detained. To assist them in their decision-making process, they may be provided with a machine learning model that predicts the risk of recidivism; i.e., the risk that the defendant will re-offend. Notably, the final decision of the judge determines the real-world consequences, not the model's prediction. Hence, if fairness is a requirement, it is not sufficient to consider the output of the model; you also need to consider how the predictions are used by the judges.

In many real-world systems, several machine learning models are deployed at the same time at different points in the decision-making process. Unfortunately, a system of components that seem fair in isolation does not automatically imply a fair system, i.e. \textit{Fair + Fair $\neq$ Fair} {footcite:p}`Dwork2020`.

To avoid the framing trap, we need to ensure that the way we frame our problem and evaluate our solution includes all relevant components and actors of the sociotechnical system.

## The Portability Trap

A system that is carefully designed for a particular context cannot always be directly applied in a different context. Taking an existing solution and applying it in a different situation without taking into account the differences between the two contexts is known as the portability trap. A shift in domain, geographical location, time, or even the nature of the decision-making process all impact the suitability of a system. For example, a voice recognition system optimized for speakers with an Australian accent may fail horribly when deployed in the United States. Similarly, the expectations of a good manager have changed considerably in the past few decades, including a stronger need for soft skills. A model trained on annual reviews in the 1960s will likely not be suitable to make predictions for current managers. The portability trap goes beyond performance issues due to differences in the data distribution. It also considers differences in social norms and actors. For example, a chatbot optimized to formulate snarky replies may be considered funny on a gaming platform, but inappropriate or even offensive in a more formal context, such as a website for loan applications. To avoid falling into the portability trap, we need to consider whether our problem understanding adequately models the social and technical requirements of the actual deployment context.

## The Formalism Trap

In order to use machine learning, you need to formulate your problem in a way that a mathematical algorithm can understand. This is not a straightforward task: there are usually many different ways to measure something. Some may be more appropriate than others. You fall into the formalism trap when your formalization does not adequately take into account the context in which your model will be used. For example, machine learning problem formulations often simplify the decision space to a very limited set of actions {footcite:p}`Mitchell2018`. In lending, the decision space of the machine learning model may consist of two options: reject or accept. In reality, there may be many more actions available, such as recommending a different type of loan.

The formalism trap is closely related to the statistical concept of construct validity: how well does the formalization measure the construct of interest? Business objectives often involve constructs such as "employee quality" or "creditworthiness" that cannot be measured directly {footcite:p}`Jacobs2019`. In such cases, data scientists may use a proxy variable instead. Every variable in a data set is the result of a decision on how a particular construct can be measured on a computer-readable scale. For example, Netflix has chosen to measure viewers' quality judgments with likes, rather than the more commonly used 1 - 5 star rating {footcite:p}`Dobbe2018`.

Not all variables measure the intended construct equally well. For example, income measures the construct socioeconomic status to some degree but does not capture other factors such as wealth and education {footcite:p}`Jacobs2019`. A mismatch between the choice of target variable and the actual construct of interest can be detrimental to fairness goals. In particular, fairness concerns can arise when the measurement error introduced by the choice of formalization differs across groups. For example, you may be interested in predicting crime, but only have access to a subset of all criminal activity: arrest records. In societies where arrest records are the result of racially biased policing practices, the measurement error will differ across racial groups. Similarly, {footcite:t}`obermeyer2019` found that due to unequal access to healthcare, historically less money has been spent on caring for African-American patients compared to Caucasian patients. Consequently, a system that used healthcare costs as a proxy for true healthcare needs systematically underestimated the needs of African-American patients.

Issues of construct validity are especially complex for social constructs such as race and gender. Almost paradoxically, measuring sensitive characteristics can introduce bias. In industry and academia, it is common to "infer" these characteristics from observed data, such as facial analysis {footcite:p}`Jacobs2019`. This can be problematic because such approaches often fail to acknowledge that social constructs are inherently contextual, may change over time, and are multidimensional. For example, when talking about race, one may be referring to somebody's racial identity (i.e., self-identified race), their observed race (i.e., the race others believe them to be), their phenotype (i.e., racial appearance), or even their reflected race (i.e., the race they believe others assume them to be). Which dimension you measure will influence the conclusions you can draw (see {footcite:t}`Hanna2020` for a more detailed account).

To avoid falling into the formalism trap, data scientists should take into account whether the problem formulation handles (social) constructs in a way that matches the intended deployment context. To mitigate threats to construct validity, ideally, multiple measures are collected, especially for complex constructs.

## The Ripple Effect Trap

Introducing a machine learning model in a social context may affect the behavior of other actors in the system and, as a result, the context itself. This is known as the ripple effect trap. There are several ways in which a social context may change due to the introduction of new technology. First, the introduction of a new technology might be used to argue for or reinforce power, which can change an organization's dynamics. For example, management may purchase software for monitoring workers, reinforcing the power relationship between management and subordinates. Second, the introduction of a prediction system may cause reactivity behavior. For example, people might attempt to game an automated loan approval system by dishonestly filling out their data in the hope of a more favorable outcome. Third, a system that was developed for a particular use case, may be used in unintended, perhaps even adversarial, ways. To avoid falling into the ripple effect trap, it is important to consider whether the envisioned system predictably changes the context.

## The Solutionism Trap

The possible benefits that machine learning solutions can bring can be very exciting. Unfortunately, machine learning is not the answer to everything (\textit{what?!}). The belief that every problem has a technological solution is referred to as solutionism. We fall into the solutionism trap when we fail to recognize machine learning is not the right tool for the problem at hand.

The solutionism trap is closely related to the optimism bias. This is a cognitive bias that causes people to overestimate the likelihood of positive events and underestimate the likelihood of negative events. In the context of algorithmic systems, optimism bias occurs when policymakers or developers are overly optimistic about a system's benefits while underestimating its limitations and weaknesses. In particular, people might overestimate the objectiveness of data and algorithmic systems. If this happens, the system's goals, development, and outcomes might not be sufficiently scrutinized, which can result in systematic harm.

There are several reasons why machine learning may not be the right tool to solve a problem. In some scenarios, it may not be possible to adequately model the context using automated data collection. For example, consider eligibility for social welfare benefits in the Netherlands. Although the criteria for eligibility are set in the law, some variables, e.g. living situation, are difficult to measure quantitatively. Moreover, the Dutch legal system contains the possibility to deviate from the criteria due to compelling personal circumstances. It is impossible to anticipate all context-dependent situations in advance. As a result, machine learning may not be the best tool for this job. In other scenarios, machine learning may be inappropriate because it lacks human connection. For example, consider a person who is hospitalized. In theory, it may be possible to develop a robot nurse who is perfectly capable of performing tasks such as inserting an IV or washing the patient. However, the patient may also value the genuine interest and concern of a nurse -- in other words, a human connection, something a machine learning model cannot (or even should not) provide. Furthermore, there may be cases where machine learning is simply overkill. For example, you may wonder whether spending several months on optimizing a deep learning computer vision system to predict the dimensions of items in your online shop is a better approach than simply asking the person who puts the item on the website to fill out the dimensions.

To avoid falling into the solutionism trap`, it is useful to consider machine learning as a means to an end. In other words, rather than asking "can we use machine learning", ask, "how can we solve this problem?" and then consider machine learning as one of the options.

## Concluding Remarks

Understanding, flagging, and avoiding abstraction traps are an important step towards machine learning systems that are not only fair in theory, but also in practice.

```{tip}
When considering designing a new (fair-)ML solution, determine...

1. Whether machine learning is the right tool for this problem in the first place (_solutionism trap_).
2. The possible unintended consequences of introducing a machine learning model in the existing
sociotechnical sytstem (_ripple effect trap_)
3. Whether the way you quantify variables and metrics makes sense (_formalism trap_)
4. Whether the requirements reflect the actual deployment context (_portability trap_)
5. Whether you have included all the relevant parts of the sociotechnical system in your model and
evaluation (_framing trap_)

```

## References

```{footbibliography}

```
