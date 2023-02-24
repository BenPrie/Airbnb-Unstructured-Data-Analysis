# Imports...
import pandas as pd
import numpy as np
import re
from datetime import datetime
import datetime as dt
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


# Primary function for computing success scores for given data.
def compute_scores(listing_directory, review_directory, review_rate=0.72, latest_date=datetime.today(), min_price=1, max_price=10000):
    '''
    This function will read the data from listing_directory and review_directory
    to generate the success score for each listing id.
    The file at each of these directories is expected to be the listings and 
    review CSV files provided by Inside Airbnb (unzipped).
    These directory variables should be strings.

    The review rate is a somewhat arbitrary decision. It refers to the estimated
    percentage of Airbnb users that leave a review after their stay. The Airbnb
    CEO, in his biased viewpoint, offers 72% as his data-driven answer.
    The given review rate must be in the interval [0, 1]

    The latest date is used to determine how recent reviews are.
    The given latest data must be a datetime object.
    '''

    # Get the raw textual data for reviews.
    review_data = pd.read_csv(review_directory)[['listing_id', 'comments', 'date']]

    # Convert string dates into datetime objects.
    review_data.date = review_data.date.apply(lambda x : datetime.strptime(x, '%Y-%m-%d'))

    # 'Fix' the format of the reviews (i.e. remove HTML markers).
    review_data.comments = review_data.comments.apply(fix_list_of_strings)

    # Remove non-string reviews and reviews shorter than two characters.
    review_data = review_data[review_data.comments.apply(type) == str]
    review_data = review_data[review_data.comments.apply(len) > 1]

    # Find overall (weighted) review sentiment for each listing...

    # Group reviews by their listing.
    grouped_review_data = review_data.groupby('listing_id')
    reviewed_listing_ids = grouped_review_data.groups.keys()

    sentiments = []
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

    sentiment_data = pd.DataFrame({'id' : reviewed_listing_ids, 'sentiment' : sentiments})

    # Get the relevant data for estimating success.
    success_data = pd.read_csv(listing_directory)[['id', 'price', 'minimum_nights_avg_ntm', 'number_of_reviews_ltm']]

    # Transform price data...

    # Convert prices to floats.
    success_data.price = success_data.price.apply(lambda x : float(re.sub(',', '', x[1:])))

    # Drop data where price is outside of the given range.
    success_data = success_data[(success_data.price >= min_price) & (success_data.price <= max_price)]

    # Apply log transformation.
    success_data['log_price'] = np.log(success_data.price)

    # Add in sentiment data.
    success_data = success_data.merge(sentiment_data, on='id')

    # Compute the success scores for each listing (where all data is available)...

    probabilities = []
    successes = []

    for i in range(len(success_data)):
        listing = success_data.iloc[i]

        # If we lack data for computing occupacncy rate, fuck it off.
        if (not type(listing.minimum_nights_avg_ntm) == np.float64) or (not type(listing.number_of_reviews_ltm) == np.float64):
            probabilities.append(np.NaN)
            successes.append(np.NaN)

        else:
            # Following from Inside Airbnb's "San Francisco Model"...

            estimated_bookings = listing.number_of_reviews_ltm / review_rate
            occupied_days = estimated_bookings * listing.minimum_nights_avg_ntm
            probability_of_rental = min(365, occupied_days) / 365
            probabilities.append(probability_of_rental)

            # Compute the success.
            success = probability_of_rental * listing.log_price * listing.sentiment
            successes.append(success)

    return_data = pd.DataFrame({
        'id' : success_data.id,
        'log_price' : success_data.log_price,
        'rental_probability' : probabilities,
        'weighted_average_sentiment' : success_data.sentiment,
        'success_score' : successes
    })

    return return_data