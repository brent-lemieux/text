import string


def preprocess_text(text):
    """Preprocess the text."""
    # Remove punctuation.
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove excess whitespace and return.
    return " ".join(text.split())
