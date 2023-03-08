# Imports as always...
import pandas as pd
import numpy as np
import re
from datetime import datetime
from textblob import TextBlob


# Helper function for 'fixing' a given string.
def fix_string(text):
    # Ignore non-string text (e.g. NaN values).
    if not(type(text) == str):
        return text

    # Use regex expression to remove '< ... >'.
    fixed = re.sub('<.*?>', ' ', text)

    # Crush whitespace.
    fixed_again = ' '.join(fixed.split())

    return fixed_again


# Modification for handling lists of strings.
def fix_list_of_strings(texts):
    # Ignore non-string text (e.g. NaN values).
    if not(type(texts) == np.ndarray):
        return texts

    # Apply fix_string to each element.
    fixes = []
    for text in texts:
        fixes.append(fix_string(text))

    return fixes