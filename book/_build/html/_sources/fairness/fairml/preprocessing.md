(fairml_pre_processing)=

# Pre-processing Algorithms

Pre-processing algorithms make adjustments to the data to mitigate the unfairness of the downstream machine learning model. Pre-processing techniques vary greatly in complexity, ranging from simple techniques that can be executed manually to more complex optimization schemes. In this section, we will cover two pre-processing approaches: relabeling and representation learning.

```{note}
The goal of this section is **not** to present a comprehensive overview of the fairness-aware machine learning techniques proposed in the algorithmic fairness literature. Instead, these examples allow you to develop an intuition of the workings, (dis)advantages, and limitations of the fairness-aware machine learning paradigm.
```

(relabeling)=

## Relabeling

The relabeling technique was first introduced by {footcite:t}`kamiran2012data`. The authors set out to learn a classifier that optimizes predictive performance, without reproducing the association between a sensitive feature $A$ and a target variable $Y$. The authors propose to pre-process the data set to remove any undesirable association between $A$ and $Y$, which can be viewed as an attempt at mitigating [historical bias](historical_bias). To this end, the relabeling technique assigns different labels (i.e., class) to instances in the training data such that the base rates of sensitive groups in the pre-processed data are equal. In particular, the relabeling technique changes labels of some instances that belong to the group with a lower base rate from positive to negative ('demotion'), and the same number of instances belonging to the group with a higher base rate from negative to positive ('promotion'). In this way, the overall class distribution in the training data is maintained.

Randomly selecting instances for relabeling likely results in a loss of predictive performance of a classifier trained on the pre-processed data. For example, relabeling a negative instance with an extremely low probability of belonging to the positive class almost certainly reduces the predictive performance of the classifier. In contrast, a negative instance that has a similar probability of belonging to either the positive or negative class could be given the benefit of the doubt.

Following this intuition, instances closest to the decision boundary are selected for relabeling. A ranker $R$ is learned on the training data, which estimates the probability of an instance belonging to the positive class. Using the ranker, instances are sorted according to their score. The highest-scoring negative instances of the group with a low base rate are up for promotion (i.e., relabeled from negative to positive), while the lowest-scoring positive instances of the group with a high base are up for demotion (i.e., relabeled from positive to negative). Relabeling is performed greedily until the base rates are equal between sensitive groups.

```{admonition} Relabeling

{footcite:t}`kamiran2012data` restrict their approach to a single binary sensitive attribute $A$ and a binary classification problem with target variable $Y$, where a positive prediction ($\hat{Y}=1$) corresponds to the desirable output for data subjects, i.e., receiving a benefit or avoiding a burden.

The input of the algorithm is labeled dataset $D = \{X, Y\}$ and sensitive feature $A$.

1. Compute the base rate, $P(Y=1 \mid A = a)$, for each sensitive group $a \in A$. Determine which of the sensitive groups has the lower base rate and vice versa.
    * $A=a$ : the group with a low base rate (i.e., $P(Y=1 \mid A = a) < P(Y=1 \mid A = a')$);
    * $A=a'$ : the group with a high base rate.
2. Learn a ranker $R$ over dataset $D$.
3. Compute the confidence score, $x(R)=P(Y=1 \mid x)$, for all instances in the training dataset ($x \in D$).
4. Identify relabeling candidates
    * Promotion candidates: $pr := \{ x \in D \mid A = a, Y=0 \}$, ordered **descending** w.r.t. $x(R)$.
    * Demotion candidates: $dem := \{ x \in D \mid A = a', Y=1 \}$, ordered **ascending** w.r.t. $x(R)$.
5. Compute the number of instances that must be relabeled to achieve equal base rates.
Let $P_{a} := P(Y=1 \mid A = a)$ and $P_{a'} := P(Y=1 \mid A = a')$ correspond to the ground-truth base rates of group $a$ and $a'$ respectively. Furthermore, let $n_{a} := |\{ x \in D \mid x(A) = a \}|$ and $n_{a'} := |\{ x \in D \mid x(A) = a' \}|$ correspond to the number of instances that belong to group $a$ and $a'$ respectively. Then, the number of instances that must be relabeled to acieve equal base rates is equal to:
$$M = \frac{(P_{a'} - P_{a}) \times n_{a} \times n_{a'}}{N}$$
If $M$ is not a whole number, the authors propose to round it up.
6. Relabel instances:
   - Relabel the top-$M$ of $pr$ from $Y=0$ to $Y=1$ ('promote');
   - Relabel the top-$M$ of $dem$ from $Y=1$ to $Y=0$ ('demote').

```

Let's have a look at an example. Consider the toy data set displayed in {numref}`toydataset`. Assuming `nationality` is considered a sensitive feature and a positive prediction (i.e., $\hat{Y}=1$) corresponds to a benefit, we can determine which labels would be changed as follows.

```{table} Toy dataset to exemplify the relabeling technique. $x(R) = P(Y=1 \mid x)$ represents the confidence score assigned by the ranker that instance $x$ belongs to the positive class.
:name: toydataset

|     | Nationality  | Highest Degree | Job Type   |  $Y$  | $x(R)$    |
| --- | ------------ | -------------- | ---------- | :-:   | :----: |
| 1   | citizen      | university     | board      |  1    |  .99   |
| 2   | citizen      | high school    | board      |  1    |  .90   |
| 3   | citizen      | university     | education  |  1    |  .92   |
| 4   | noncitizen   | university     | healthcare |  1    |  .72   |
| 5   | noncitizen   | none           | healthcare |  0    |  .44   |
| 6   | noncitizen   | high school    | board      |  0    |  .09   |
| 7   | citizen      | university     | education  |  0    |  .66   |
| 8   | citizen      | none           | healthcare |  1    |  .66   |
| 9   | noncitizen   | high school    | education  |  0    |  .02   |
| 10  | citizen      | university     | board      |  1    |  .92   |

```

First, we need to determine the base rates of sensitive groups. Considering the $Y$ column, 5 of the 6 `citizen` instances belong to class $Y=1$, while only 1 of the 4 `noncitizen` instances belong to class $Y=1$. Consequently, low-scoring positive instances of `citizen` are up for demotion, while high-scoring negative instances of `noncitizen` are up for promotion.

Now we need to determine how many instances ought to be relabeled to achieve equal base rates. We have $P_a = 1/4$, $n_a = 4$, $P_{a'} = 5/6$, $n_{a'}=6$, and $N = 10$. As such, $M = (5/6 - 1/4)*4*6/10 = 1.4$ instances much be relabeled. Following the procedure as suggested by the authors, we round up to 2.

In this case, therefore, instances 8 and 2 will be relabeled from positive to negative and instances 5 and 6 will be relabeled from negative to positive. In the pre-processed dataset, the base rates for `citizen` and `noncitizen` are 3/6 and 3/4 respectively. Note that by rounding up the number of relabeled instances the overall class distribution remains the same (6/10) but now the base rate of `noncitizen` is higher than `citizen`.

% ## Feature transformation

(representation_learning)=

## Representation Learning

Relabeling makes explicit adjustments to the training data to obscure the association between a sensitive feature and the target variable. Computer science researchers have also proposed more abstract pre-processing approaches, in which the problem is formulated as an optimization task to identify a new representation of the data. The first of this kind is the representation learning approach proposed by {footcite:t}`zemel2013learning`.

{footcite:t}`zemel2013learning` approach unfairness mitigation as an optimization problem of finding a good representation of the data with two - competing - goals: (1) encoding the data as well as possible, while (2) obscuring information regarding sensitive group membership. The representation of choice is a probabilistic mapping from instances in the original dataset to a set of prototypes. A prototype can be viewed as an artificial instance that is representative of a cluster of instances. A prototype is represented by a vector $v_k$, which is in the same space as the instances $x \in X$. The learned mapping assigns a probability to each of the instances in the dataset of mapping onto each of the prototypes $k \in Z$ ({numref}`representationlearning1`). Based on these probabilities, the final transformation of the data set stochastically assigns each input example to one of the prototypes.

```{figure} ../../figures/representationlearning1.svg
:name: representationlearning1
:width: 500px
:align: center
:target: none

Representation learning aims to learn a probablistic mapping from instances $x \in X$ to a set of prototypes $Z$, in which each instance is assigned a probability of mapping to each of the prototypes $k \in Z$.
```

The probabilistic mapping is defined based on the distance between an instance and a prototype, quantified via a distance function $d(x,v_k)$ ({numref}`representationlearning`). Typical choices of distance functions are Euclidean distance or cosine distance. The distances are transformed to probabilities via the softmax function, i.e., $P(Z=k \mid x) = \frac{\exp(-d(x,v_k))}{\sum_{k=1}^K exp(-d(x,v_j))}$.

```{figure} ../../figures/representationlearning.svg
:name: representationlearning
:width: 300px
:align: center
:target: none

Each instance is represented by a dot. Prototype $k$ can be viewed as an artificial instance, representative of a cluster of instances. Each prototype is represented by a vector $v_k$, which is in the same space as instances $x$. Distance function $d(x,v_k)$ quantifies the similarity between an instance and a prototype.
```

Learning the mapping $X \rightarrow Z$ is formulated as an optimization task with three goals.

1. Information regarding sensitive group membership is lost by requiring $P(Z=k \mid A=a) =P(Z=k \mid A=a')$. That is, the probability of mapping upon a particular prototype should be independent of sensitive group membership.
2. Information in $X$ is retained as best as possible;
3. The mapping $X \rightarrow Z \rightarrow Y$ is close to the mapping $f : X \rightarrow Y$.

Each of these aims is translated to a term in the objective function that is used to learn the representations. For the exact details of the formulation of the objective function, we refer the interested reader to {footcite:t}`zemel2013learning`.

## References

```{footbibliography}

```
