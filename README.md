# **Text Classification for High School Exam Questions**

## Highlights:
1. This is a **multi-class text classification (document classification)** problem.
2. The purpose of this project is to classify High School Exam Questions into some classes and **the number of classes is related to the data set**.

## demands:
1. You can solve this problem with a variety of **machine learning** algorithms.
2. The evaluation method is mainly based on **precision and recall**.


## Data:
Chinese exam questions of high school.


## Train-test split:
In order to unify the standard, we use the questions whose ID end with 9 as the test set and the rest as the train set.

## Evaluation:
```ruby
def count_precision_recall_at_k(y_pred, y_true, k):
    """
    y_pred: [[ 1.3315865   0.71527897 -1.54540029 -0.00838385  0.62133597 -0.72008556]]
    y_true: [[0 0 1 1 0 0]
    """
    y_indices = y_pred.argsort()[:, -k:][:, ::-1]
    pre = 0.0
    rec = 0.0
    for i in range(len(y_true)):
        intersec_true = 0
        for j in y_indices[i]:
            intersec_true += y_true[i][j]
        true_total_count = np.count_nonzero(y_true[i] == 1)
        pred_total_count = len(y_indices[i])
        pre += intersec_true*1.0/pred_total_count
        rec += intersec_true*1.0/true_total_count
    return pre/len(y_true), rec/len(y_true)
```


*These baselines are the results of two different algorithms.*

## Reference:
[Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)</br>
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)</br>
[A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820)</br>
[Very Deep Convolutional Networks for Text Classification](https://arxiv.org/abs/1606.01781)</br>
[Hierarchical Attention Networks for Document Classification](http://www.aclweb.org/anthology/N16-1174)

