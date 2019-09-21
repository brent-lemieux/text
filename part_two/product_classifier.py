"""Contains ProductClassifier class.

Classifies product receipt codes using fastText from Facebook Open Source --
https://fasttext.cc/docs/en/python-module.html.

Example:
>>> from product_classifier import ProductClassifier

>>> clf = ProductClassifier(dim=50, maxn=6, minn=2)
>>> clf.fit('./data/labeled.csv')
>>> clf.label_csv('./data/unlabeled')

"""

from collections import defaultdict
from datetime import datetime
import json
import os

import numpy as np
import pandas as pd

import fasttext

from training_data import format_train_file
from utils import preprocess_text


class ProductClassifier(object):
    """
    Classify products codes using fastText from Facebook Open Source.

    Methods:
        fit: trains the model with the option to save to disk
        cross_validate: splits the data into k-folds, trains on each training
            fold, and computes metrics on each validation fold
        label_csv: takes an unlabeled csv, appends predicted labels, and writes
            back to disk
        predict: takes text, or an array of texts, and returns predicted labels
            and the associated probablity of each predicted label

    Attributes:
        clean_text_ (bool, default False): apply the `preprocess_text` function
            to the `text_column` if True
        model_ (fastText obj): trained fastText model; only available after
            running methodds fit or load_model
        model_info_ (dict): trained model information; includes model
            hyper parameters and preprocessing instructions;
            only available after running methodds fit or
            load_model
        cv_accuracy_ (list): contains overall accuracy of each cross-validation
            fold; only available after running method cross_validate
        cv_precision_recall_ (dict): contains precision, recall, and f1 scores
            for each category and each cross-validation fold; only available
            after running method cross_validate
    """

    def __init__(self, clean_text=False, lr=.1, dim=100, ws=5, epoch=5,
                 minCount=1, minCountLabel=1, minn=0, maxn=0, neg=5,
                 wordNgrams=1, loss='softmax', bucket=2000000,
                 lrUpdateRate=100, t=.0001):
        """
        Constructor for ProductClassifier class.

        Arguments:
            clean_text_ (bool, default False): apply the `preprocess_text`
                function to the `text_column` if True
            model_ (fastText obj): trained fastText model
            lr (float, default 0.1): learning rate
            dim (int, default 100): size of word vectors
            ws (int, default 5): size of the context window
            epoch (int, default 5): number of epochs
            minCount (int, default 1): minimal number of word occurences
            minCountLabel (int, default 1): minimal number of label occurences
            minn (int, default 0): min length of char ngram
            maxn (int, default 0): max length of char ngram
            neg (int, default 5): number of negatives sampled
            wordNgrams (int, default 1): max length of word ngram
            loss (str, default softmax): loss function {ns, hs, softmax, ova}
            bucket (int, default 2000000): number of buckets
            lrUpdateRate (int, default 100): change the rate of updates for
                learning rate
            t (float, default 0.0001): sampling threshold
        """
        self.clean_text_ = clean_text
        self.model_ = None
        self.hyper_params_ = {
            'lr': lr,
            'dim': dim,
            'ws': ws,
            'epoch': epoch,
            'minCount': minCount,
            'minCountLabel': minCountLabel,
            'minn': minn,
            'maxn': maxn,
            'neg': neg,
            'wordNgrams': wordNgrams,
            'loss': loss,
            'bucket': bucket,
            'lrUpdateRate': lrUpdateRate,
            't': t
        }

    def fit(self, labeled_csv, model_key='dev', text_column='item_text',
            label_column='cat', output_prefix=None, save_model=False):
        """
        Method to update the existing model, or build a new model. Saves and
        returns model object.

        Arguments:
            labeled_csv (str): /path/to/labeled.csv
            model_key (str, default dev): key for identifying different model
                versions in model and model_info filepaths
            text_column (str, default item_text): column name of text to be
                classified
            label_column (str, default cat): column name of categorical label
            output_prefix (str, default None): prefix for train.txt filename
            save_model (bool, default False): save model object if True
        """
        if not os.path.exists(labeled_csv):
            raise Exception(
                'Labeled filepath: {} does not exist'.format(labeled_csv)
            )
        # Format training data into .txt file per fastText requirements.
        train_filepath = format_train_file(
            labeled_csv, text_column=text_column, label_column=label_column,
            clean_text=self.clean_text_, output_prefix=output_prefix
        )
        # Train and save model.
        self.model_ = fasttext.train_supervised(train_filepath,
                                                **self.hyper_params_)
        if save_model:
            self.save_model(model_key)
        return self.model_

    def save_model(self, model_key):
        """
        Helper function for fit.

        Saves model and model information with model_key as identifier. Model
        information contains model hyper-parameters and preprocessing
        intstructions.
        """
        # Save model.
        if not os.path.exists('./models'):
            os.mkdir('./models')
        model_filepath = './models/{}_model.bin'.format(model_key)
        self.model_.save_model(model_filepath)
        # Save model_info.
        model_info_filenpath = './models/{}_info.json'.format(model_key)
        self.model_info_ = {
            model_key: {
                'hyper_params': self.hyper_params_,
                'clean_text': self.clean_text_
            }
        }
        json.dump(self.model_info_, open(model_info_filenpath, 'w'))

    def cross_validate(self, labeled_csv, n_cv_splits=3, log_results=False,
                       text_column='item_text', label_column='cat',
                       output_prefix=None):
        """
        Method to cross-validate the model for the given hyper parameters.

        Arguments:
            labeled_csv (str): /path/to/labeled.csv
            n_cv_splits (int, default 3): number of data splits for
                cross-validation
            log_results (bool, default False): log cross-validation run results
                if True
            text_column (str, default item_text): column name of text to be
                classified
            label_column (str, default cat): column name of categorical label
            output_prefix (str, default None): prefix for train.txt and
                validation.txt filenames
        """
        if not os.path.exists(labeled_csv):
            raise Exception(
                'Labeled filepath: {} does not exist'.format(labeled_csv)
            )
        # Write fastText inputs for training and validation and get filepaths.
        cv_filepaths = format_train_file(
            labeled_csv, text_column=text_column,
            label_column=label_column, cross_validation=True,
            n_cv_splits=n_cv_splits, clean_text=self.clean_text_,
            output_prefix=output_prefix
        )
        # Train and validate each cross-validation fold.
        precision_recalls = []
        accuracies = []
        for train_filepath, val_filepath in cv_filepaths:
            model = fasttext.train_supervised(train_filepath,
                                              **self.hyper_params_)
            fold_result = model.test_label(val_filepath)
            precision_recalls.append(fold_result)
            accuracies.append(model.test(val_filepath)[1])
        # Set cross-validation metrics.
        self.cv_accuracy_ = accuracies
        self.cv_precision_recall_ = self.parse_results(precision_recalls)
        # Log cross-validation results.
        if log_results:
            self.log_results()

    def parse_results(self, results):
        """
        Helper function for cross_validate.

        Parses and combines model results of each validation split.
        """
        master_results = defaultdict(list)
        # Iterate through each fold in k-fold cross-validation restults.
        for fold_result in results:
            # Iterate through each categories results.
            for category, category_dict in fold_result.items():
                category_name = category.replace('__label__', '')
                # Append category-metrics to master_results.
                for metric_name, score in category_dict.items():
                    key_name = '{}_{}'.format(category_name, metric_name)
                    master_results[key_name].append(score)
        return master_results

    def log_results(self):
        """
        Helper function for cross_validate.

        Logs cross-validation results of each run, along with hyper-parameters
        to track model tuning.
        """
        run_log = {
            'hyper_params': self.hyper_params_,
            'precision_recall': self.cv_precision_recall_,
            'overall_accuracy': self.cv_accuracy_
        }
        # Load log if exists.
        if os.path.exists('results_log.json'):
            model_log = json.load(open('./results_log.json', 'r'))
        # Create new log if else.
        else:
            model_log = []
        # Append this runs data to the log.
        run_key = len(model_log) + 1
        model_log.append({run_key: run_log})
        json.dump(model_log, open('./results_log.json', 'w'))

    def label_csv(self, unlabeled_csv, model_key=None, text_column='item_text',
                  output_prefix=None):
        """
        Label unlabeled csv file with fastText, save output, and return output
        filepath.

        Read in the unlabeled csv file, make predictions on the `text_column`,
        and append the predicted labels. Write the file with label predictions.

        Arguments:
            unlabeled_csv (str): /path/to/unlabeled.csv -- data to label
            model_key (str, default dev): key for identifying different model
                versions in model and model_info filepaths
            text_column (str, default item_text): column name of text to be
                classified
            output_prefix (str, default None): prefix for output filename
        """
        if not os.path.exists(unlabeled_csv):
            raise Exception(
                'Unlabeled filepath: {} does not exist'.format(unlabeled_csv)
            )
        # If model_key is passed, attempt to load the model specified by key.
        if model_key:
            self.load_model(model_key)
            # Set self.clean_text_ based on model_info.
            self.clean_text_ = self.model_info_[model_key]['clean_text']
            print('clean_text = {} per {} model'.format(self.clean_text_,
                                                        model_key))
        # Check to see if the model is already trained.
        elif self.model_:
            pass
        else:
            raise Exception('Model not trained and no model_key provided.')
        # Read in unlabeled data set for labeling.
        try:
            df = pd.read_csv(unlabeled_csv)
        # If default encoding fails, try ISO-8859-1.
        except UnicodeDecodeError:
            df = pd.read_csv(unlabeled_csv, encoding="ISO-8859-1")
        # Get predictions for text column and format output.
        predictions, probas = self.predict(df[text_column].values.tolist())
        predictions = [
            label[0].replace('__label__', '') for label in predictions
        ]
        df['cat'] = predictions
        if not output_prefix:
            output_prefix = ''
        output_csv = './data/{}predicted_labels.csv'.format(output_prefix)
        df.to_csv(output_csv)
        return output_csv

    def predict(self, text):
        """
        Predicts class of text and returns the label prediction and the model
        probability of the predicted label.

        If multiple text items are passed, the method returns a tupel
        containing two arrays -- array one contains predicted labels and array
        two contains predicted label probabilities.

        Arguments:
            text (str or list): text to be classified.
        """
        # If self.clean_text_ apply preprocess_text function according to type.
        if self.clean_text_:
            if type(text) == str:
                text = preprocess_text(text)
            else:
                text = [preprocess_text(item_text) for item_text in text]
        # Predict and return text label and probability.
        return self.model_.predict(text)

    def load_model(self, model_key):
        """
        Load model and model info using the model_key.

        Arguments:
            model_key (str): key for identifying the model version to load
        """
        # Load the model object.
        model_filepath = './models/{}_model.bin'.format(model_key)
        if not os.path.exists(model_filepath):
            raise Exception(
                'Model filepath: {} does not exist.'.format(model_filepath)
            )
        self.model_ = fasttext.load_model(model_filepath)
        # Load the model info dict.
        model_info_filepath = './models/{}_info.json'.format(model_key)
        if not os.path.exists(model_info_filepath):
            raise Exception(
                'Model info filepath: {} does not exist.'.format(
                    model_info_filepath
                )
            )
        self.model_info_ = json.load(open(model_info_filepath, 'r'))
