"""Label unlabeled.csv with ProductClassifier."""
from product_classifier import ProductClassifier


def main():
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
    # Log cross-validation metrics.
    clf.cross_validate('./data/labeled.csv', n_cv_splits=3, log_results=True)
    # Train model using labeled data.
    clf.fit('./data/labeled.csv')
    # Use model to label unlabeled data.
    clf.label_csv('./data/unlabeled.csv')


if __name__ == '__main__':
    main()
