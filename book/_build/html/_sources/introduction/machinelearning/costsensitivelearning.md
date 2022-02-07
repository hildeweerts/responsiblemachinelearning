# Cost-Sensitive Learning
In practice, we often associate different costs with different types of mistakes. Cost-sensitive learning is a way to explicitly take into account different valuations of mistakes in learning and model selection.

<!-- ## Optimizing the Decision Threshold
Based on the ROC curve, we can choose the decision threshold that is likely to give us the true positive rate and false positive rate that are useful for your use case. As with any optimization, you can determine the expected performance on new data by evaluating the performance of the chosen threshold on the test set. It is also possible to tune decisionthreshold using, for example, grid search.

*Coming soon*: example of how we can choose the decision threshold based on a validation set.

## Cost-Sensitive Learning Algorithms
It is often also possible to directly take into account misclassification cost during learning. In particular, it may be possible to weigh instances of a particular class differently in the algorithm's loss function. For example, {py:func}`sklearn.linear_model.LogisticRegression` has a `class_weight` parameter that allows to adjust the weights for different classes.

*Coming soon*: example of algorithm that allows to specificy different costs -->