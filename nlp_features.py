# Imports, as always...
import pandas as pd
import numpy as np
from datetime import datetime

import miscellaneous_helpers as mh

from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer


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


def create_features_revised(listings_directory, reviews_directory, latest_date=datetime.now()):
    '''
    This is a revised version of the create_features function.
    Instead of TextBlob, this uses NLTK to create positive, negative, neutral, and
    compound components of sentiment, rather than a single polarity score.
    This should make it more robust against more ambigious text.

    This revision also skips keyword extraction -- it was not useful.
    '''

    # Get the raw textual data for each listing.
    raw_data = pd.read_csv(listings_directory)[['id', 'name', 'description', 'neighborhood_overview', 'host_about']]

    # 'Fix' the text.
    raw_data.name = raw_data.name.apply(mh.fix_string)
    raw_data.description = raw_data.description.apply(mh.fix_string)
    raw_data.neighborhood_overview = raw_data.neighborhood_overview.apply(mh.fix_string)
    raw_data.host_about = raw_data.host_about.apply(mh.fix_string)

    nlp_data = pd.DataFrame()

    analyser = SentimentIntensityAnalyzer()

    title_neg = []
    title_neu = []
    title_pos = []
    title_compound = []
    description_neg = []
    description_neu = []
    description_pos = []
    description_compound = []
    neighborhood_overview_neg = []
    neighborhood_overview_neu = []
    neighborhood_overview_pos = []
    neighborhood_overview_compound = []
    host_about_neg = []
    host_about_neu = []
    host_about_pos = []
    host_about_compound = []

    for i in range(len(raw_data)):
        datum = raw_data.iloc[i]

        if type(datum.values[1]) == str:
            title_analysis = analyser.polarity_scores(datum.values[1])
            title_neg.append(title_analysis['neg'])
            title_neu.append(title_analysis['neu'])
            title_pos.append(title_analysis['pos'])
            title_compound.append(title_analysis['compound'])
        else:
            title_neg.append(0)
            title_neu.append(0)
            title_pos.append(0)
            title_compound.append(0)

        if type(datum.values[2]) == str:
            description_analysis = analyser.polarity_scores(datum.values[2])
            description_neg.append(description_analysis['neg'])
            description_neu.append(description_analysis['neu'])
            description_pos.append(description_analysis['pos'])
            description_compound.append(description_analysis['compound'])
        else:
            description_neg.append(0)
            description_neu.append(0)
            description_pos.append(0)
            description_compound.append(0)
        
        if type(datum.values[3]) == str:
            neighborhood_overview_analysis = analyser.polarity_scores(datum.values[3])
            neighborhood_overview_neg.append(neighborhood_overview_analysis['neg'])
            neighborhood_overview_neu.append(neighborhood_overview_analysis['neu'])
            neighborhood_overview_pos.append(neighborhood_overview_analysis['pos'])
            neighborhood_overview_compound.append(neighborhood_overview_analysis['compound'])
        else:
            neighborhood_overview_neg.append(0)
            neighborhood_overview_neu.append(0)
            neighborhood_overview_pos.append(0)
            neighborhood_overview_compound.append(0)
        
        if type(datum.values[4]) == str:
            host_about_analysis = analyser.polarity_scores(datum.values[4])
            host_about_neg.append(host_about_analysis['neg'])
            host_about_neu.append(host_about_analysis['neu'])
            host_about_pos.append(host_about_analysis['pos'])
            host_about_compound.append(host_about_analysis['compound'])
        else:
            host_about_neg.append(0)
            host_about_neu.append(0)
            host_about_pos.append(0)
            host_about_compound.append(0)

    nlp_data['id'] = raw_data.id
    nlp_data['title_neg'] = title_neg
    nlp_data['title_neu'] = title_neu
    nlp_data['title_pos'] = title_pos
    nlp_data['title_compound'] = title_compound
    nlp_data['description_neg'] = description_neg
    nlp_data['description_neu'] = description_neu
    nlp_data['description_pos'] = description_pos
    nlp_data['description_compound'] = description_compound
    nlp_data['neighborhood_overview_neg'] = neighborhood_overview_neg
    nlp_data['neighborhood_overview_neu'] = neighborhood_overview_neu
    nlp_data['neighborhood_overview_pos'] = neighborhood_overview_pos
    nlp_data['neighborhood_overview_compound'] = neighborhood_overview_compound
    nlp_data['host_about_neg'] = host_about_neg
    nlp_data['host_about_neu'] = host_about_neu
    nlp_data['host_about_pos'] = host_about_pos
    nlp_data['host_about_compound'] = host_about_compound

    # Compute the perceived review scores.
    review_sentiments = compute_review_scores_revised(reviews_directory, latest_date)

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


def compute_review_scores_revised(reviews_directory, latest_date=datetime.now()):
    '''
    Revised version of compute_review_scores.
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

    perceived_review_neg = []
    perceived_review_neu = []
    perceived_review_pos = []
    perceived_review_compound = []

    analyser = SentimentIntensityAnalyzer()

    for id in reviewed_listing_ids:
        group = grouped_review_data.get_group(id)

        negs = []
        neus = []
        poss = []
        compounds = []
        weights = []
        for i in range(len(group)):
            entry = group.iloc[i]

            # Convert the date into a number of days that have elapsed since the each review was made.
            elapsed = (latest_date - entry.date).days

            # The weight of each listing will be in the interval [0, 1].
            weight = 1 / max(1, elapsed)
            weights.append(weight)

            # Determine the polarity of the review.
            review_analysis = analyser.polarity_scores(entry.comments)
            negs.append(review_analysis['neg'])
            neus.append(review_analysis['neu'])
            poss.append(review_analysis['pos'])
            compounds.append(review_analysis['compound'])

        # Scale weights to maintain the proportions, but make the maximum weight 1.
        scaled_weights = list(map(lambda x : x / max(weights), weights))

        # Compute weighted polarities.
        perceived_review_neg.append(np.average(negs, weights=scaled_weights))
        perceived_review_neu.append(np.average(neus, weights=scaled_weights))
        perceived_review_pos.append(np.average(poss, weights=scaled_weights))
        perceived_review_compound.append(np.average(compounds, weights=scaled_weights))

    # Return the sentiment scores.
    return pd.DataFrame({
        'id' : reviewed_listing_ids, 
        'perceived_review_neg' : perceived_review_neg,
        'perceived_review_neu' : perceived_review_neu,
        'perceived_review_pos' : perceived_review_pos,
        'perceived_review_compound' : perceived_review_compound
    })