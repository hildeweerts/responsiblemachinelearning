(introduction)=
# Responsible Machine Learning

With the advent of large-scale data collection, the toolkit of a data scientist has proven to be a powerful way to make products and processes faster, cheaper, and better. Many data science applications make use of {term}`machine learning` algorithms: algorithms that build mathematical models by ‘learning’ from data. Nowadays, machine learning models are integrated in many computer systems: from music recommendations to automated fraud detection, facial recognition systems, and personalized medicine assistants. These systems can provide benefits, but are not without risks.

A responsible data scientist understands how machine learning models might be harmful and how the risks can be mitigated. This online book provides a practical introduction to the nascent field of responsible machine learning.

## What do I mean with *Responsible Machine Learning*?
So what is a 'responsible' approach? Generally speaking, responsibility is a *‘duty or obligation to take care of something’*. Taking responsibility involves actively avoiding that something 'bad; happens or increasing the probability that something 'good' happens. What can be considered 'right' or 'wrong' is the central question of ethics and has occupied philosophers for many centuries.

As machine learning systems are increasingly deployed in all kinds of applications, several incidents have shown in what ways machine learning systems can have negative consequences. Machine learning models can inherit existing prejudices embedded in society, which can result in discrimination. The increasing complexity of machine learning systems can lead to opaque decision-making systems.

<!-- And the increasing power of the organizations who deploy these systems raises questions about accountability. -->

I have organized these lecture notes along two key moral values: fairness and transparency. Although there exists some overlap, each theme emphasizes a different aspect of a responsible approach to machine learning.

### Fairness

{term}`Fairness` is a moral value that concerns treatment of behavior that is *just* and *free from discrimination*. Machine learning models, particularly classifiers, are specifically designed to discriminate between cases. As with any decision-making process, these distinctions can be undesirable from an ethical perspective or unlawful if they disproportionately affect people on the basis of sensitive characteristics such as gender, race, religion, age, and sexual orientation.

In recent years, several incidents have shown that machine learning systems can inherit and amplify social biases embedded in society. In 2016, investigative journalists from Propublica found that COMPAS, a decision support tool for assessing the likelihood of a defendant becoming a recidivist, wrongly labeled African-American defendants as recidivists at much higher rates than white Americans {footcite:p}`compas2017`. Although concerns regarding the fairness of algorithmic decision-making systems is not new, Propublica’s article sparked an increased interest in the field.

Although math and numbers may seem objective, machine learning models are not value neutral. They are the result of many design choices that embed the values of their developers: which data is collected, what metrics are used to evaluate our models, and which problems do we decide to tackle in the first place? The risk of unfairness is not limited to adversarial actors - even a well-intentioned data scientist has their blind spots.

### Transparency

As a moral value, {term}`transparency` can be defined as the *degree of openness that allows others to understand what actions are performed*. In the context of machine learning, an important dimension of transparency is the extent to which we can understand a model’s prediction-generating process. This is known as {term}`interpretable machine learning` or {term}`explainable machine learning`.

In some cases, the best performing models are complex models such as ensembles or deep neural networks. As the complexity of models increases, it generally becomes more difficult for humans to understand their behavior. In many contexts, it can be valuable or even imperative to understand why a machine learning model makes certain predictions. For example, machine learning practitioners might use explanations to understand where the model fails and how it might be improved. 

<!-- ### Accountability

Previously, I have defined responsibility as a duty to take care of something. Responsibility can also be defined as being *accountable* for something. {term}`Accountability` considers being held responsible for one’s actions, typically after something 'bad' has happened. Due to the apparent complexity of algorithmic systems, organizations may try to divert blame to the algorithm: *“oh, it’s just the algorithm.”* Algorithmic accountability is the idea that an institution should be held accountable for the use, design, and decisions of an algorithmic system. It involves taking adequate measures to comply with ethical principles or legal regulations, including detailed documentation and clear procedures for appealing decisions.

An important tool for fostering accountability is auditing, in which the development process, usage, and impact of an algorithmic system are closely inspected - either through internal procedures or by an external third party. -->

## For who is this book?

The intended audience of this book are (undergraduate) computer science students, practitioners, and researchers interested in responsible machine learning. It is expected that you are familiar with machine learning in general and supervised machine learning in particular. This includes the high-level workings of basic classification algorithms, model selection approaches, and evaluation metrics. To get you up to speed, I will go over some of the basic concepts in the chapter [Machine Learning Preliminaries](ml_preliminaries).

## What is covered in this book?

With the increasing usage of machine learning in real-world applications, we have also seen more examples of how machine learning can go wrong. As a result, the interest in a more responsible approach to computer science has surged. At one end of the spectrum, the field has been productive in generating principles, guidelines, and frameworks for more 'ethical' artificial intelligence. However, many of these principles are too broad to guide the daily practice of a data scientist or machine learning engineer. At the other end of the spectrum, many technical solutions have been proposed, which can forego the real-world context of machine learning applications.

In this book, I have tried to balance these extremes by setting out a practical approach that acknowledges the sociotechnical context in which machine learning applications are used.

The techniques, challenges, and considerations discussed in these lecture notes mostly involve applications based on supervised learning. Nonetheless, the sociotechnical approach that is encouraged throughout the lecture notes can be applied to any analytics project.

Of course, there is much more to cover than what reasonably fits into one book. In addition to the values of fairness and explainability, other important requirements are, for example, technical safety, privacy, sustainability. Additionally, there are many adjacent topics such as moral philosophy, accessible product design, and organizational best practices. Although organizational structures, including efforts such as ethics committees and programs for diversity and inclusion, are crucial for operationalizing responsible machine learning these are not the main focus of these lecture notes.

## Navigating this book

The rest of this book is organized as follows: 

* [Machine Learning](ml_preliminaries) contains a quick recap of relevant machine learning concepts.
* [Fairness](intro_fairness) dives into the topic of algorithmic fairness, including techniques to discover and mitigate undesirable discrimination that have been proposed in the compute science literature. 
* [Explainable Artificial Intelligence](intro_xai) covers several techniques for creating interpretable models and explaining black-box machine learning models.

```{note}

   Developing machine learning applications is subjective: it will require you to make many decisions, which often involves making trade-offs between competing criteria. I believe that taking responsibility requires one to make these decisions explicit. Given the great variety of norms and values that exist in the world, I would like to acknowledge that some of the ideas discussed in this book may reflect perspectives you do not agree with. The goal of this book is not to enforce a particular moral framework or political point-of-view. Instead, I aim to provide you with the knowledge and tools to identify trade-offs and make substantiated decisions.

   This book summarizes many of the things I have learned so far on my journey in the field of responsible machine learning. Nevertheless, I am but a person with my own limited perspective. In particular, I would like to acknowledge that as an able-bodied white women that was born and raised in the Netherlands, I lack lived experience that is the daily reality of many people in marginalized communities. My background is primarily technical and I am still learning from all great scholars in this interdisciplinary field. I hope you will too.
```

## References

```{footbibliography}
```
