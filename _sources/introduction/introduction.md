# Responsible Data Science

With the advent of large-scale data collection, the toolkit of a data scientist has proven to be a powerful way to make products and processes faster, cheaper, and better. Many data science applications make use of {term}`machine learning` algorithms: algorithms that build mathematical models by `learning' from data. Nowadays, machine learning models are integrated in many computer systems: from music recommendations to automated fraud detection, facial recognition systems, and personalized medicine assistants. These systems can provide benefits, but are not without risks. 

A responsible data scientist understands how machine learning models might be harmful and how the risks can be mitigated. This online book provides a practical introduction to the nascent field of responsible machine learning.

## What is Responsible Data Science?
So what is a 'responsible' approach? Generally speaking, responsibility is a 'duty or obligation to take care of something'. From an ethical perspective, taking responsibility involves actively avoiding that something 'bad' happens or increasing the probability that something 'good' happens. Now, what can be considered 'right' or 'wrong' has been the central question of ethics for centuries.

As machine learning systems are increasingly deployed in all kinds of applications, several incidents have shown in what ways machine learning systems can have bad consequences. Machine learning models can inherit existing prejudices embedded in society, which can result in discrimination. The increasing complexity of machine learning systems can lead to opaque decision-making systems. And the increasing power of the organizations who deploy these systems raises questions about accountability.

I have organized these lecture notes along three key moral values: fairness, transparency, and accountability. Although there exists some overlap, each theme emphasizes a different aspect of a responsible approach to machine learning.

### Fairness
Fairness is a moral value that concerns treatment of behavior that is just and free from discrimination. Machine learning models, particularly classifiers, are specifically designed to discriminate between cases. As with any decision-making process, these distinctions can be undesirable or unlawful if they disproportionately affect people on the basis of sensitive characteristics such as gender, race, religion, age, and sexual orientation.

In recent years, several incidents have shown that machine learning systems can inherit and amplify social biases embedded in society. In 2016, investigative journalists from Propublica found that COMPAS, a decision support tool for assessing the likelihood of a defendant becoming a recidivist, wrongly labeled African-American defendants as recidivists at much higher rates than white Americans {cite:p}`compas2017`. Although concerns regarding the fairness of algorithmic decision-making systems is not new, Propublica's article sparked an increased interest in the field.

Although math and numbers may seem objective, machine learning models are not value neutral. They are the result of many design choices that embed the values of their developers: which data is collected, what metrics are used to evaluate our models, and what problems do we decide to tackle in the first place? Consequently, the risk of unfairness is not limited to nefarious actors - even a well-intentioned data scientist has their blind spots.

### Transparency
As a moral value, transparency can be dedined as the degree of openness that allows others to understand what actions are performed. In the context of machine learning, an important dimension of transparency is the extent to which we can understand a model's prediction-generating process. This is known as 'interpretable' or 'explainable' machine learning.

In some cases, the best performing models are complex models such as deep neural networks and ensembles. As the complexity of models increases, it generally becomes more difficult for humans to understand their behavior. In many contexts, it can be valuable or even imperative to understand why a machine learning model makes certain predictions. For example, machine learning practitioners might use explanations to understand where the model fails and how it might be improved. Moreover, in regulated fields such as banking and health care, the ability to justify decisions is often mandated by law. In this case, explainability can be closely related to accountability.

### Accountability
Previously, I have defined responsibility as a duty to take care of something. Responsibility can also be defined as being accountable for something. This second definition considers being held responsible after something `bad' has happened. Due to the apparent complexity of algorithmic systems, organizations may try to divert blame to the algorithm: *"oh, it's just the algorithm."* Algorithmic accountability is the idea that an institution should be held accountable for the use, design, and decisions of an algorithmic system. It involves taking adequate measures to comply with ethical principles or legal regulations, including detailed documentation and clear procedures for appealing decisions.  

An important tool for fostering accountability is auditing, in which the development process, usage, and impact of an algorithmic system are closely inspected. This can be done either through internal procedures or by an external third party. 

## For who is this book?
The intended audience of this book are (undergraduate) data science students, practitioners, and researchers interested in responsible data science. It is expected that you are familiar with machine learning in general and supervised machine learning in particular. This includes the high-level workings of basic classification algorithms, model selection approaches, and evaluation metrics. Additionally, I expect readers to have some basic knowledge on normative ethical frameworks and terminology. To get you up to speed, I will go over some of the basic concepts in the chapter [Machine Learning](mlpreliminaries).

## What is covered in this book?
With the increasing usage of machine learning in real-world applications, I have also seen more examples of how machine learning can go wrong. As a result, the interest in a more responsible approach to data science has surged. At one end of the spectrum, the field has been productive in generating principles, guidelines, and frameworks for more 'ethical' artificial intelligence. However, many of these principles are too broad to guide the daily practice of a data scientist. At the other end of the spectrum, many technical solutions have been proposed. Although these efforts are laudable, these solutions often forego the real-world context of machine learning applications. 

In this book, I have tried to balance these extremes by setting out a practical approach that acknowledges the sociotechnical context in which machine learning applications are used.

The techniques, challenges, and considerations discussed in these lecture notes mostly involve applications based on supervised learning. Nonetheless, the sociotechnical approach that is encouraged throughout the lecture notes can be applied to any analytics project.

Of course, there is much more to cover than what reasonably fits into one book. In addition to the values of fairness, explainability, and accountability, the EU High Level Expert Group on AI has identified additional key requirements of trustworthy artificial intelligence {cite:p}`intelligence2020assessment`: (1) human agency and oversight, (2) technical robustness and safety, (3) privacy and data governance, and (4) sustainability. Additionally, there are many adjacent topics such as moral philosophy, accessible product design, and organizational best practices. Although I wholeheartedly believe in the importance of organizational changes, including efforts such as ethics committees and programs for diversity and inclusion, these are not the main focus of these lecture notes.

## Navigating this book
The rest of this book is organized as follows:
* Chapter [Machine Learning](mlpreliminaries) contains a quick recap of relevant concepts in machine learning.
* Chapters [Group Fairness](groupfairness), [Individual Fairness](individualfairness), and [Counterfactual Fairness](counterfactualfairness) dive into the topic of algorithmic fairness, including techniques to discover and mitigate undesirable discrimination that have been proposed in the computer science literature.
* [Explainable Machine Learning](introxai) covers several techniques for creating interpretable models and explaining black-box machine learning models.
* [Algorithmic Accountability and Auditiong](introaccountability) introduces algorithmic accountability and provides an introduction to internal and external audits of algorithmic decision-making systems.

```{note}
Developing machine learning applications is not an objective task: it will require you to make many decisions, which often involves making trade-offs between competing criteria. This is a subjective task. I believe that taking responsibility requires making these decisions explicit. At times, this will require ethical decision-making. Given the great variety of norms and values that exist in the world, I would like to acknowledge that some of the ideas discussed in this book may reflect perspectives you do not agree with or values you do not personally consider important. This is okay. The goal of this book is not to enforce a particular moral framework or political point-of-view. Instead, I aim to provide you with the knowledge and tools to identify trade-offs and make substantiated decisions.

This book summarizes many of the things I have learned so far on my journey in the field of responsible data science. Nevertheless, I am also but a person, with my own limited perspective. In particular, I would like to acknowledge that as an able-bodied white women that was born and raised in the Netherlands, I lack lived experience that is the daily reality of many people in marginalized communities. Additionally, my background is primarily technical. I am still learning. I hope you will too.
```

```{bibliography}
:filter: docname in docnames
```