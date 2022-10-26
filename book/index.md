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

# An Introduction to Responsible Machine Learning

by *Hilde Weerts*

```{code-cell} ipython3
---
tags: [remove-cell]
---
from datetime import date
from myst_nb import glue
glue("today", date.today().strftime('%Y-%m-%d'))
```

Last updated: {glue:text}`today`

```{note}
This book is still very much under development. If you have suggestions, questions, or comments, please feel free to open an issue on the [Github repository](https://github.com/hildeweerts/responsibleml).
```

---

With the advent of large-scale data collection, the toolkit of a data scientist has proven to be a powerful way to make products and processes faster, cheaper, and better. Many data science applications make use of {term}`machine learning` algorithms: algorithms that build mathematical models by 'learning' from data. Nowadays, machine learning models are integrated in many computer systems: from music recommendations to automated fraud detection, facial recognition systems, and personalized medicine assistants. These systems can provide benefits, but are not without risks.

A responsible data scientist understands the potential harm of machine learning models and how to mitigate the risks. This online book provides a practical introduction to the nascent field of responsible machine learning. The goal of this book is to provide a practical approach, building a bridge between philosophical, social, and technical perspectives.

`````{grid} 1 2 2 2
:gutter: 3

````{grid-item-card} Fairness
Data-driven systems can inherit the existing prejudices embedded in society, resulting in systematic discrimination or other harms.

```{button-ref} intro_fairness
:expand:
:color: primary
:outline:

learn more
```

````

````{grid-item-card} Transparency
As the complexity of machine learning models increase, it becomes more difficult for humans to understand, assess, and justify their behavior.

```{button-ref} intro_xai
:expand:
:color: primary
:outline:

learn more
```

````
`````

---

**Citing this book**

To cite this book, please use the following bibtex entry:

```{code-block}
@book{weerts2022,
  title = {An Introduction to Responsible Machine Learning},
  author = {Hilde Weerts},
  year = {2022}
  url = {https://hildeweerts.github.io/responsiblemachinelearning/}
}
```

--- 
This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg