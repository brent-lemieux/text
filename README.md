# Answers

## Part One - Debugging
There are several issues and inefficiencies in the original `inner_median` function.

**Issues**
* The code is written to select the middle item from the `intersection` list, or the average of the two middle items if the list has an even length. However, `intersection` is never sorted. Because of this, any item that ends up in the middle of `intersection` will be selected, not necessarily the median value.
* The `intersection` list is created using a nested loop. The code loops through `x`, and for each item in `x`, loops through `y`, appending the item in `x` to `intersection` when it is equal to the item in `y`. This creates duplicate values in `intersection`, which will cause issues when calculating the median of the intersection in many cases.

**Inefficiencies**
* The code relies on for loops and indexing into lists when the `numpy` methods `intersect1d` and `median` will handle the problem much more efficiently. Aside from executing faster, using the `numpy` methods makes the function much easier to understand.

## Part Two - Text Classification
The goal of this project is to categorize products from grocery store receipts into one of the five groups:
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

## Methodology and Approach
My approach to this problem was to:
1. Explore the data, think about the problem, and read up on how others have solved similar problems.
2. Select a model.
3. Iterate on different modeling decisions, including hyper-parameter selection and text preprocessing -- testing each iteration with cross-validation.

The goal of this project is to categorize product receipt names for grocery store items. To solve this problem, I chose to use [fastText](https://fasttext.cc/), a text-classification algorithm from Facebook Open Source.

**fastText overview**
The algorithm represents words, character n-grams, and word n-grams, as vectors. The vectors are called word embeddings. It then averages the word/n-gram embeddings for each document, or product receipt names in this case, to obtain the items feature vector. Once the feature vectors are computed, the algorithm uses multinomial logistic regression to classify the product receipt names.

I selected fastText for several reasons.
1. Accuracy and speed
2. Easy to use Python API
3. Subword embeddings

### Accuracy and speed
According to fastText's creators, "Our experiments show that our fast text classifier fastText is often on par with deep learning classifiers in terms of accuracy, and many orders of magnitude faster for training and evaluation."

I found the algorithm's results to be quite good. With a little bit of tuning, the model correctly classified an items category 87% of the time in cross-validation tests. Training the model took only a few seconds. More on results later.

### Easy to use Python API
With limited time to work on the problem, I knew I needed a solution that was easy to implement. fastText is easy to use without sacrificing much (if any) predictive power.

### Subword embeddings
As noted in the instructions, "Grocery store receipts often contain abbreviated, and even cryptic, language representing the products that have been sold." Mainly because of the sometimes cryptic language, I thought embeddings at the character or subword level would improve the quality of predictions. Luckily, fastText implements word, subword, and word n-gram embeddings. This theory proved accurate. When allowing for subword embeddings of 2 or more characters, as well as word and word n-gram embeddings, model performance improved significantly.

For example, using subword embeddings, the model can relate different yogurt products like `OIKOS YOGURT`, `OIKOSYOG`, and `YOPLT YOGU`.

## Model Results
I was very happy with the model results given the limited time. After a little bit of time spent hyper-parameter tuning, the average cross-validation accuracy of the model was 87%, meaning the model selected the appropriate category 87% of the time. There was only a small amount of variance in the results from fold to fold in the cross-validation tests.

The model performed very well on beverages, dairy, and produce. All of these categories averaged F1 scores above 85%. Baking (76%) and beer (62%) did not do as well. F1 score is the harmonic mean of precision and recall, which are explained below in the **Dairy Results Analysis** section.

### Dairy Results Analysis
* Precision - 89%; meaning when the model classified the item as dairy, it was correct 89% of the time.
* Recall - 88%; meaning of all the dairy items in the validation set, the model identified 88% of them as dairy.

## Next Steps
* Test each function and method using ``pytest``
* Write a function to grid-search different modeling decisions like hyper-parameters and text preprocessing
* Apply deep learning methods to see if predictive power can be improved

## Production
I would choose to implement this data product as a real-time API. This would open up the potential for the product to add more value to the user. For instance, category predictions could be used as a part of a content-based recommender system. If a user uploads their receipt today, they could immediately be shown deals for items categorically similar to those they just purchased.

### Steps required to put this solution in production
1. Communicate with the application engineers to understand what data is available in real-time and the format of that data.
2. Develop an endpoint for the model in SageMaker and integrate it with API Gateway.
3. Write tests that call the API to ensure that it takes data in, and returns data, in the required format.
4. Perform load testing to ensure that the API remains performant at the upper bounds of expected concurrent requests.
5. Run integration tests, or work with the application engineers to ensure that the API works as expected with the application.
6. Go to production.


### Steps to ensure quality over time
There are many methods I would use to ensure the model API maintains a certain level of quality over time. I would write software to automatically assess the following, and notify me of any changes above a defined threshold so I could investigate further:
1. Assess whether the distribution of predicted categories shifts over time. This may require accounting for the seasonality of certain product categories, i.e. produce.
2. Assess distribution shifts at a more micro level. Do predicted category distribution shifts occur at single stores or groups of stores?
3. If the data product is used in a way that the customer interacts with, assess the quality of user interaction over time. For example, if the data product is used as a part of a content-based recommender system, track click and buy rates on the products recommended from this system over time. If the rates fall, it may be due to a lapse in quality in the categorization model.

### Steps to update the production model given a change in category taxonomy
1. While working on a solution to predict the new category taxonomy, maintain the solution currently in production.
2. Assuming that the change in category taxonomy comes with newly labeled data, repeat the modeling steps described above with the new data.
3. Repeat the steps required for the initial production implementation.
4. Switch application traffic to the new version. Maintain the old version until confident the new version is performing as expected in production.
