# Imports, as always...
import pandas as pd
import numpy as np

import success_metric as sm

from textblob import TextBlob


# Helper function for keyword extraction.
def extract_n_keywords(text, n):
    # If text is not in fact text, return NaN.
    if type(text) != str:
        return np.NaN

    assessments = TextBlob(text).sentiment_assessments.assessments

    # If there are no keywords, return NaN.
    if not len(assessments):
        return np.NaN
    
    # Sort the keywords by the magnitude of their polarity (descending).
    assessments = sorted(assessments, key=lambda x : abs(x[1]), reverse=True)

    # Drop the sentiment scores of the keywords.
    keywords = list(map(lambda x : x[0], assessments))

    return keywords[:n]


# Primary function for creating NLP features.
def create_features(listing_directory, keyword_limit=5, title_keyword_limit=2):
    '''
    This function will read the data from listing_directory to create NLP features.
    The file at this directories is expected to be the listings CSV files provided by Inside 
    Airbnb (unzipped). This should be a string.

    The keyword limit allows us to specify the maximum number of keywords to be extracted for
    each source of raw textual data.
    '''

    # Get the raw textual data for each listing.
    raw_data = pd.read_csv(listing_directory)[['id', 'name', 'description', 'neighborhood_overview', 'host_about']]

    # 'Fix' the text.
    raw_data.name = raw_data.name.apply(sm.fix_string)
    raw_data.description = raw_data.description.apply(sm.fix_string)
    raw_data.neighborhood_overview = raw_data.neighborhood_overview.apply(sm.fix_string)
    raw_data.host_about = raw_data.host_about.apply(sm.fix_string)

    # Perform sentiment analysis to obtain polarity and subjectivity scores...
    # Perform keyword extraction to obtain (at most) the n most polarising keywords.

    nlp_data = pd.DataFrame()

    nlp_data['id'] = raw_data.id

    nlp_data['title_polarity'] = raw_data.name.apply(
        lambda x : TextBlob(x).sentiment.polarity if type(x) == str
                                                  else np.NaN
    )

    nlp_data['title_subjectivity'] = raw_data.name.apply(
        lambda x : TextBlob(x).sentiment.subjectivity if type(x) == str
                                                      else np.NaN
    )

    nlp_data['title_keywords'] = raw_data.name.apply(
        lambda x : extract_n_keywords(x, title_keyword_limit)
    )

    nlp_data['description_polarity'] = raw_data.description.apply(
        lambda x : TextBlob(x).sentiment.polarity if type(x) == str
                                                  else np.NaN
    )

    nlp_data['description_subjectivity'] = raw_data.description.apply(
        lambda x : TextBlob(x).sentiment.subjectivity if type(x) == str
                                                      else np.NaN
    )

    nlp_data['description_keywords'] = raw_data.description.apply(
        lambda x : extract_n_keywords(x, keyword_limit)
    )

    nlp_data['neighborhood_overview_polarity'] = raw_data.neighborhood_overview.apply(
        lambda x : TextBlob(x).sentiment.polarity if type(x) == str
                                                  else np.NaN
    )

    nlp_data['neighborhood_overview_subjectivity'] = raw_data.neighborhood_overview.apply(
        lambda x : TextBlob(x).sentiment.subjectivity if type(x) == str
                                                      else np.NaN
    )

    nlp_data['neighborhood_overview_keywords'] = raw_data.neighborhood_overview.apply(
        lambda x : extract_n_keywords(x, keyword_limit)
    )

    nlp_data['host_about_polarity'] = raw_data.host_about.apply(
        lambda x : TextBlob(x).sentiment.polarity if type(x) == str
                                                  else np.NaN
    )

    nlp_data['host_about_subjectivity'] = raw_data.host_about.apply(
        lambda x : TextBlob(x).sentiment.subjectivity if type(x) == str
                                                      else np.NaN
    )

    nlp_data['host_about_keywords'] = raw_data.host_about.apply(
        lambda x : extract_n_keywords(x, keyword_limit)
    )

    return nlp_data