# Imports as always...
import pandas as pd
import numpy as np
import re
from datetime import datetime
from textblob import TextBlob

import miscellaneous_helpers as mh


# Primary function for computing success scores for given data.
def compute_scores(listing_directory, review_rate=0.72, min_price=1, max_price=10000):
    '''
    This function will read the data from listing_directory to generate the success score for each listing id.
    We also set a minimum and maximum price, mainly to allow us to prevent erroneous data.

    listing_dictionary will be a path (as a string) to the listings.csv file.
    review_rate is the estimated (or known) ratio of the number of reviews to the number of bookings.
        Brian Chesky, in his almighty wisdom, has claimed this is around 72%, so that will be the default.
    min_price and max_price will be numerical values denoting the minimum price and maximum price.
        Any listings with an advertised price outside of their defined range will be disregarded.

    The return will be a dataframe with columns "id" and "success_score".
    '''

    # Get the relevant data for estimating success.
    success_data = pd.read_csv(listing_directory)[['id', 'price', 'minimum_nights_avg_ntm', 'number_of_reviews_ltm', 'review_scores_rating']]

    # Remove listings lacking all the necessary data for success computation.
    success_data = success_data.dropna()

    # Transform price data...

    # Convert prices to floats.
    success_data.price = success_data.price.apply(lambda x : float(re.sub(',', '', x[1:])))

    # Drop data where price is outside of the given range.
    success_data = success_data[(success_data.price >= min_price) & (success_data.price <= max_price)]

    # Apply log transformation.
    success_data['log_price'] = np.log(success_data.price)

    # Compute the success scores for each listing (where all data is available)...

    successes = []

    for i in range(len(success_data)):
        listing = success_data.iloc[i]

        # Following from Inside Airbnb's "San Francisco Model"...

        estimated_bookings = listing.number_of_reviews_ltm / review_rate
        occupied_days = estimated_bookings * listing.minimum_nights_avg_ntm
        probability_of_rental = min(365, occupied_days) / 365

        # Compute the success.
        success = probability_of_rental * listing.log_price * (listing.review_scores_rating / 5)
        successes.append(success)

    return_data = pd.DataFrame({'id' : success_data.id, 'success_score' : successes})

    return return_data