# Imports.
import pandas as pd
from datetime import datetime
import numpy as np
import ast

import success_metric as sm
import nlp_features as nlp

# Primary function for creating data for a given city.
def city_data_generation(city, datasets_directory, latest_date=datetime.now(), add_image_keywords=False):
    '''
    To use this function, we assume the naming convention '{"listing"/"reviews"}_{city name}.csv'

    This function will take the given city and datasets directory to find the raw data.
    From this data, it will create the master dataframe from which we will train and test our models.

    city is expected to be a string corresponding to the suffix of the relevant csv files.
    datasets_directory is expected to be the path (as a string) to the folder in which the data is stored.
    Note: the datasets_directory argument should not be succeeded by a '/' (this function will add one in).

    latest_date is used to determine recency with reviews, which impacts review sentiment and hence success score.
    e.g. for city "A" and datasets_directory "./data", the function will look for "./data/listings_A.csv" and "./data/reviews_A.csv".
    
    add_image_keywords (optional) tells us whether to look for an image_keywords.csv file and add in its data.
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

    # Merge the raw_data with the NLP features, retaining all listings.
    master_data = raw_data.merge(nlp_features, on='id', how='outer')

    # Add in image keywords if wanted.
    if add_image_keywords:
        # Read in the keyword data.
        keyword_data = pd.read_csv(datasets_directory + '/image_keywords_' + city + '.csv')
        keyword_data.rename({'keywords' : 'images_keywords', 'confindences' : 'images_confidences'}, axis=1, inplace=True)

        # Merge the master data with the keyword data.
        master_data = master_data.merge(keyword_data, on='id', how='outer')

    # Merge the master data with the success data.
    master_data = master_data.merge(success_scores, on='id', how='outer').reset_index()

    # Write the master data to a new file.
    master_data.to_csv('{}/master_{}.csv'.format(datasets_directory, city))


# Function for computing mean success scores for keywords
def mean_keyword_scores(success_scores, keyword_directory):
    '''
    This function will compute the mean success score of all listings containing each keyword.
    The return will be a dictionary of the form {keyword : mean score}. It will not be sorted.

    success_scores will be a dataframe containing listing 'ids' and their corresponding 'success_score'.
    keyword_directory will be a path to the image_keywords.csv file containing a list of ids and their corresponding keywords (and confidences).

    Warning: this function will not work if the success_scores dataframe does not contain columns named 'ids' and 'success_score'.
    Likewise, the keyword_directory csv file must have 'ids', 'keywords' and 'confindences' (yes, mispelled).
    '''

    # Read in the keyword data.
    keyword_data = pd.read_csv(keyword_directory)

    # Merge with the success data.
    data = success_scores[['id', 'success_score']].merge(keyword_data).dropna()

    keyword_scores = {}

    # Loop through each entry...
    for i in range(len(data.index)):
        entry = data.iloc[i]

        # Convert the keyword and confidence objects to usable lists of strings and floats.
        keywords = ast.literal_eval(entry.keywords)
        confidences = ast.literal_eval(entry.confindences)

        # Loop through each keyword in the entry...
        for j in range(len(keywords)):
            # Track every score it achieves.
            if keywords[j] in keyword_scores.keys():
                # Note the score in proportion to the confidence, such that a low confidence constitutes a lower magnitude score.
                keyword_scores[keywords[j]].append(entry.success_score * (confidences[j] / 100))
            
            else:
                keyword_scores[keywords[j]] = [entry.success_score * (confidences[j] / 100)]

    # Take a mean average of each keywords scores.
    for keyword in keyword_scores.keys():
        keyword_scores[keyword] = np.mean(keyword_scores[keyword])

    return keyword_scores