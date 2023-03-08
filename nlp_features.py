# Imports, as always...
import pandas as pd
import numpy as np
from datetime import datetime

import miscellaneous_helpers as mh

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
def create_features(listings_directory, reviews_directory, keyword_limit=5, title_keyword_limit=2, latest_date=datetime.now()):
    '''
    This function will read the data from listing_directory to create NLP features.
    The file at this directories is expected to be the listings CSV files provided by Inside 
    Airbnb (unzipped). This should be a string.

    The keyword limit allows us to specify the maximum number of keywords to be extracted for
    each source of raw textual data.
    '''

    # Get the raw textual data for each listing.
    raw_data = pd.read_csv(listings_directory)[['id', 'name', 'description', 'neighborhood_overview', 'host_about']]

    # 'Fix' the text.
    raw_data.name = raw_data.name.apply(mh.fix_string)
    raw_data.description = raw_data.description.apply(mh.fix_string)
    raw_data.neighborhood_overview = raw_data.neighborhood_overview.apply(mh.fix_string)
    raw_data.host_about = raw_data.host_about.apply(mh.fix_string)

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

    # Compute the perceived review scores.
    review_sentiments = compute_review_scores(reviews_directory, latest_date)

    # Merge the review data with the rest.
    nlp_data = nlp_data.merge(review_sentiments, on='id', how='outer')

    return nlp_data


# Function for computing perceived review sentiment.
def compute_review_scores(reviews_directory, latest_date=datetime.now()):
    '''
    This function will perform sentiment anaylsis on the reviews of each listing to
    compute a "perceived review sentiment score".
    This will be a weighted average of computed sentiment polarities, where weight
    is given by the recency of the review with regard to the latest date considered.

    reviews_directory will be the path (as a string) to the reviews.csv file.
    latest_date will be the date to compute recency with regard to (as a datetime
    object) -- recency will be the number of days between the review and latest date.

    The return will be a dataframe with columns "id", "perceived_review_sentiment",
    "min_review_sentiment", and "max_review_sentiment".
    '''

    # Get the review data.
    review_data = pd.read_csv(reviews_directory)[['listing_id', 'date', 'comments']]

    # Convert string dates into datetime objects.
    review_data.date = review_data.date.apply(lambda x : datetime.strptime(x, '%Y-%m-%d'))

    # 'Fix' the format of the reviews (i.e. remove HTML markers).
    review_data.comments = review_data.comments.apply(mh.fix_list_of_strings)

    # Remove non-string reviews and reviews shorter than two characters.
    review_data = review_data[review_data.comments.apply(type) == str]
    review_data = review_data[review_data.comments.apply(len) > 1]

    # Group reviews by their listing.
    grouped_review_data = review_data.groupby('listing_id')
    reviewed_listing_ids = grouped_review_data.groups.keys()

    sentiments = []
    minimums = []
    maximums = []

    for id in reviewed_listing_ids:
        group = grouped_review_data.get_group(id)

        polarities = []
        weights = []
        for i in range(len(group)):
            entry = group.iloc[i]

            # Convert the date into a number of days that have elapsed since the each review was made.
            elapsed = (latest_date - entry.date).days

            # The weight of each listing will be in the interval [0, 1].
            weight = 1 / max(1, elapsed)
            weights.append(weight)

            # Determine the polarity of the review.
            polarity = TextBlob(entry.comments).sentiment.polarity
            polarities.append(polarity)

        # Scale weights to maintain the proportions, but make the maximum weight 1.
        scaled_weights = list(map(lambda x : x / max(weights), weights))

        # Compute the weighted average of review sentiments.
        sentiment = np.average(polarities, weights=scaled_weights)

        sentiments.append(sentiment)
        minimums.append(min(polarities))
        maximums.append(max(polarities))

    # Return the sentiment scores.
    return pd.DataFrame({
        'id' : reviewed_listing_ids, 
        'perceived_review_sentiment' : sentiments,
        'min_review_sentiment' : minimums,
        'max_review_sentiment' : maximums
    })