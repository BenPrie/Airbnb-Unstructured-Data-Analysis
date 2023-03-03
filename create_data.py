# Imports.
import pandas as pd
from datetime import datetime

import success_metric as sm
import nlp_features as nlp

# Primary function for creating data for a given city.
def city_data_generation(city, datasets_directory, latest_date=datetime.now()):
    '''
    To use this function, we assume the naming convention '{"listing"/"reviews"}_{city name}.csv'

    This function will take the given city and datasets directory to find the raw data.
    From this data, it will create the master dataframe from which we will train and test our models.

    city is expected to be a string corresponding to the suffix of the relevant csv files.
    datasets_directory is expected to be the path (as a string) to the folder in which the data is stored.
    Note: the datasets_directory argument should not be succeeded by a '/' (this function will add one in).

    latest_date is used to determine recency with reviews, which impacts review sentiment and hence success score.

    e.g. for city "A" and datasets_directory "./data", the function will look for "./data/listings_A.csv" and "./data/reviews_A.csv".
    '''

    # Establish the full paths to the raw data.
    listings_directory = datasets_directory + '/listings_' + city + '.csv'
    reviews_directory = datasets_directory + '/reviews_' + city + '.csv'

    # Get the raw data.
    raw_data = pd.read_csv(listings_directory)
    
    # Drop the raw textual data from the raw_data.
    raw_data.drop(columns=[
        'name', 
        'description', 
        'neighborhood_overview', 
        'host_about'
    ], axis=1, inplace=True)

    # Drop irrelevant data.
    raw_data.drop(columns=[
        'listing_url', 
        'scrape_id', 
        'last_scraped', 
        'source', 
        'picture_url', 
        'host_url', 
        'host_name',
        'host_location',
        'host_thumbnail_url',
        'host_picture_url',
        'calendar_updated',
        'calendar_last_scraped'
    ], axis=1, inplace=True)

    # Generate success scores and NLP features.
    success_scores = sm.compute_scores(listings_directory, reviews_directory, latest_date=latest_date)
    nlp_features = nlp.create_features(listings_directory)

    # Merge the raw_data with the success scores and NLP features, retaining all listings.
    master_data = raw_data.merge(nlp_features, on='id', how='outer').merge(success_scores, on='id', how='outer').reset_index()

    # Write the master data to a new file.
    master_data.to_csv('{}/master_{}.csv'.format(datasets_directory, city))

    