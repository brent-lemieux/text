# Answers

## Part One - Debugging
There are a number of issues and inefficiencies in the original `inner_median` function.

**Issues**
* The code is written to select the middle item from the `intersection` list, or the average of the two middle items if the list has an even length. However, `intersection` is never sorted. Because of this, any item that ends up in the middle of `intersection` will be selected, not necessarily the median value.
* The `intersection` list is created using a nested loop. The code loops through `x`, and for each item in `x`, loops through `y`, appending the item in `x` to `intersection` when it is equal to the item in `y`. This creates duplicate values in `intersection`, which will cause issues when calculating the median of the intersection in many cases.

**Inefficiencies**
* The code relies on for loops and indexing into lists when the `numpy` methods `intersect1d` and `median` will handle the problem much more efficiently. Aside from executing faster, using the `numpy` methods makes the function much easier to understand.

## Part Two - Text Classification
The goal of this project is to categorize product receipt codes into one of the five groups:
* baking
* beer
* beverages
* dairy
* produce

## Quick Start
**Dependencies:**
* `numpy`
* `pandas`
* `fasttext`

**Example:**
```python
from product_classifier import ProductClassifier

# Set hyper parameters.
hyper_params = {
    "maxn": 8,
     "minn": 2,
     "dim": 50,
     "lr": 0.2,
     "ws": 5,
     "wordNgrams": 2
}
# Create ProductClassifier object.
clf = ProductClassifier(clean_text=False, **hyper_params)
# Cross validate model and log metrics.
clf.cross_validate('./data/labeled.csv', n_cv_splits=3, log_results=True)
# Train model using labeled data.
clf.fit('./data/labeled.csv')
# Use model to label unlabeled data.
clf.label_csv('./data/unlabeled.csv')
```

## Methodology
The goal of this project is to categorize product codes for grocery store items. To solve this problem, I chose to use (fastText)[https://fasttext.cc/], a text-classification algorithm from Facebook Open Source. The algorithm represents words, subwords, and n-grams



I selected fastText for a number of reasons.
1. Accuracy and speed
2. Easy to use Python API
3. Word and subword vector representation

**Accuracy and speed**
fastText trains text
