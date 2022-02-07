# Glossary
```{glossary}

accountability
   *(as a moral value)* Holding one responsible for one's actions, typically after harmful consequences have occurred.

accuracy
   A predictive performance metric defined as the fraction of predictions the model predicted correctly: $\frac{tp + tn}{tp+tn+fp+fn}$.

AUC
   A predictive performance metric defined as the area under the {term}`ROC curve`.

average precision
   The average {term}`precision` score over all {term}`recall` values in a {term}`precision-recall curve`.

binary classification
   A {term}`classification` problem with exactly two {term}`class`es.

calibration
   The extent to which a predicted value corresponds to the actual probability that an instance belongs to the predicted class.

calibration curve
   A curve that sets out the mean predicted probability of a model against the fraction of positives.

class
   A category of the categorical {term}`target variable` in a {term}`classification` problem.

classification
   A {term}`supervised learning` problem where the target variable is categorical.

confidence score
   The confidence of the model that an {term}`instance` belongs to a certain {term}`class`.

confusion matrix
   A matrix that denotes the performance of a machine learning model. The columns denote predicted {term}`class`es, the rows the ground0truth classes, and the cells the number of instances for which the combinatino of prediciton/ground truth class occurs.

decision threshold
   In a binary classification problem, the cut-off value of the model's predicted score at which an instance is classified as belonging to the {term}`positive class`.

explainable machine learning
   A branch of {term}`machine learning` that studies explaining machine learning models' inner workings or predictions in a way that is understandable to humans, typically concerns complex black-box models. Sometimes used interchangeably with {term}`interpretable machine learning`.

f1-score
   A predictive performance metric that is equal to the harmonic mean of {term}`precision` and {term}`recall`: $\frac{tp}{tp + \frac{1}{2}(fp+tn)}$.

fairness
   *(as a moral value)* Treatment of behavior that is just and free from discrimination.

false negative
   A positive instance that the model incorrectly predicted to be negative.

false negative rate
   A predictive performance metric defined as the fraction of {term}`false negative`s out of all instances that belong to the {term}`positive class`: $\frac{fn}{fn+tp}$.

false positive
   A negative instance that the model incorrectly predicted to be positive.

false positive rate
   A predictive performance metric defined as the fraction of {term}`false positive`s out of all instances that belong to the {term}`negative class`: $\frac{fp}{fp+tn}$.

feature
   A variable in a data set that is used to predict the {term}`target variable`.

grid search
   A {term}`hyperparameter tuning` procedure that consists of an exhaustive search over all hyperparameter settings in the search space.

hyperparameter
   A parameter of a machine learning algorithm that controls the learning process in some way and is set by the user.

hyperparameter tuning
   The process of identifying the best {term}`hyperparameter` settings for a learning task.

imbalanced data
   A dataset with skewed class proportions, i.e., the number of instances per class differs across classes. 

instance
   A record in a data set.

interpretable machine learning
   A branch of {term}`machine learning` that studies explaining machine learning models' inner workings or predictions in a way that is understandable to humans, typically concerns intrinsically interpretable models. Sometimes used interchangeably with {term}`explainable machine learning`.

k-fold cross-validation
   A model selection procedure in which the data set is split into $k$ folds. For each fold, a machine learning model is tested on the current fold and trained on the remaining $k-1$ folds. The overall performance of an algorithm is then estimated based on the performance on all $k$ folds.

loss function
   A function that formalizes the costs of different types of errors in an optimization algorithm.

machine learning
   A research field that studies the use and development of algorithms that build mathematical models by `learning' from experience or data.

negative class
   The {term}`class` in a {term}`binary classification` problem we are not the most interested in, e.g., not fraud in fraud detection.

negative predictive value
   A predictive perforance metric defined as the fraction of instances that belong to the {term}`negative class` among the predicted negative instances: $\frac{tn}{tn+fn}$.

nested cross-validation
   A model selection procedure in which {term}`k-fold cross-validation` is performed for both the split between {term}`training set` and {term}`test set` and the split between {term}`training set` and {term}`validation set`.

overfitting
   A scenario where the model relies on artefacts in the {term}`training set` that do not generalize beyond the training data.

positive class
   The {term}`class` in a {term}`binary classification` problem we are trying to detect, e.g., fraud in fraud detection.

positive predictive value
   A predictive perforance metric defined as the fraction of instances that belong to the {term}`positive class` among the predicted positive instances: $\frac{tp}{tp+fp}$. Equivalent to precision.

precision
   A predictive performance metric defined as the fraction of instances that belong to the {term}`positive class` among the predicted positive instances: $\frac{tp}{tp+fp}$.

precision-recall curve
   A curve in which the {term}`precision` of a model is set out against the {term}`recall` of the model at different {term}`decision threshold`s.

random search
   A {term}`hyperparameter tuning` procedure that consists of a search of a predefined number of randomly sampled hyperparameter settings from the search space.

recall
   A predictive performance metric defined as the fraction of instances that belong to the {term}`positive class` from all positive instances in the data set. Equivalent to the {term}`true positive rate`: $\frac{tp}{tp+fn}$.

regression
   A {term}`supervised learning` problem where the target variable is continuous.

reinforcement learning
   A subfield of {term}`machine learning` where the goal is to learn which actions to take in order to maximize a particular reward. 

repeated cross-validation
   A {term}`k-fold cross-validation` procedure that is repeated multiple times, shuffling the data differently each time.

ROC curve
   A curve in which the {term}`true positive rate` of a model is set out against the {term}`false positive rate` of the model at different {term}`decision threshold`s.

supervised learning
   A subfield of {term}`machine learning` where the goal is learn to predict a {term}`target variable` from input features.

target variable
   The variable that is to be predicted in {term}`supervised learning`.

test set
   The data that is used to estimate the performance of the machine learning model on new data *after* model selection.

training set
   The data that is used to fit the model to.

transparency
   *(as a moral value)* The degree of openness that allows others to understand what actions are performed.

true negative
   A negative instance that the model correctly predicted to be negative.

true negative rate
   A predictive performance metric defined as the fraction of {term}`true negative`s out of all instances that belong to the {term}`negative class`: $\frac{tn}{fp+tn}$.

true positive
   A positive instance that the model correctly predicted to be positive.

true positive rate
   A predictive performance metric defined as the fraction of {term}`true positive`s out of all instances that belong to the {term}`positive class`: $\frac{tp}{fn+tp}$.

underfitting
   A scenario where the model is not able to capture the underlying structure of the data.

unsupervised learning
   A subfield of {term}`machine learning` where the goal is to identify patterns in the data in absence of an explicit target variable, such as clusters of similar instances or topics of texts.

validation set
   The data that is used to estimate the performance of a machine learning for model selection.
````