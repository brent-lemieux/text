import numpy as np
import pandas as pd

from utils import preprocess_text


def format_train_file(labeled_csv, text_column='item_text', label_column='cat',
                      cross_validation=False, n_cv_splits=3,
                      clean_text=False, output_prefix=None):
    """
    Returns the train.txt filepath, or a list of (train.txt, val.txt)
    filepaths for cross-validation.

    Format the labeled .csv training file for fastText text classification
    model. `fasttext.train_supervised` takes a .txt file as an input. Each
    training example is confined to a single line in the format:
     "__label__<category here> <text here>"

     Example line: "__label__dairy OIKOS GRK YOG"

    Arguments:
        labeled_csv (str): /path/to/labeled.csv
        text_column (str, default item_text): column name of text to be
            classified
        label_column (str, default cat): column name of categorical label
        cross_validation (bool, default False): whether or not to divide the
            data into n_cv_splits
        n_cv_splits (int; default 3): number of k-fold splits for
            cross_validation; only relevant if cross_validation is True
        clean_text (bool, default False): apply the `preprocess_text` to the
            `text_column` if True
        output_prefix (str, default None): prefix for train.txt (and
            validation.txt) filenames

    Returns:
        filepaths (str list<tuples>): if cross_validation is False, returns
            filepath of training data as a str. Otherwise, returns a list of
            tuples containing training and validation filepaths for
            cross-validation -- [('train1.txt', 'val1.txt'), ...].
    """
    # Load and preprocess the text data.
    try:
        df = pd.read_csv(labeled_csv)
    # If default encoding fails, try ISO-8859-1.
    except UnicodeDecodeError:
        df = pd.read_csv(labeled_csv, encoding="ISO-8859-1")
    if clean_text:
        df[text_column] = df[text_column].map(preprocess_text)
    # Concatenate the label and text for fastText model.
    df['data'] = '__label__' + df[label_column] + ' ' + df[text_column]
    # Return cross-validation filepaths.
    if cross_validation:
        # Prevent duplicate rows from being in both train and validation sets.
        df = df.drop_duplicates([text_column, label_column])
        return cv_data_writer(df['data'], n_cv_splits, output_prefix)
    # Return train data filepath.
    return data_writer(df['data'], output_prefix=output_prefix)


def cv_data_writer(series, n_cv_splits, output_prefix):
    """
    Helper function for format_train_file. Writes cross-validation files in
    fastText format and returns a list of tuples containing the filepaths.
    """
    # Build a list of output files.
    k_fold_outputs = []
    data = split_data(series, n_cv_splits)
    data_idx = list(range(len(data)))
    for val_idx in data_idx:
        k = val_idx + 1
        # Write k-fold train data.
        train_idx = [idx for idx in data_idx if idx != val_idx]
        train_data = [data[idx] for idx in train_idx]
        train_data = np.concatenate(train_data)
        train_output = data_writer(train_data, ftype='train', index=k,
                                   output_prefix=output_prefix)
        # Write k-fold validation data.
        val_data = data[val_idx]
        val_output = data_writer(val_data, ftype='validation', index=k)
        # Append output filenames to k_fold_outputs for cross-validation.
        k_fold_outputs.append((train_output, val_output))
    return k_fold_outputs


def split_data(series, n_cv_splits):
    """
    Helper function for cv_data_writer. Returns a list of series split into
    n_cv_splits.
    """
    # Shuffle the dataframe.
    series = series.sample(frac=1)
    return np.array_split(series, n_cv_splits)


def data_writer(series, ftype='train', index=None, output_prefix=None):
    """
    Helper function for format_train_file. Write training data in fastText
    format and returns the filepath.
    """
    if not index:
        index = ''
    if not output_prefix:
        output_prefix = ''
    output_name = './data/{}{}{}.txt'.format(output_prefix, ftype, index)
    with open(output_name, 'w') as f:
        for item in series:
            f.write('{}\n'.format(item))
    return output_name
