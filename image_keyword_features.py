# Imports, as always...
import pandas as pd
import numpy as np
import random
import os
import ast
from textblob import TextBlob

import image_labels

# Primary function for creating the dictionaries of extracted keywords (with their confidence levels).
def keyword_extraction(listing_directory, destination_directory, city_name=None, picture_limit=3, confidence_threshold=0, overwrite=False):
    '''
    This function generates a list of keywords and their associated confidence levels for each listing.

    listing_directory should be the path to the listing.csv files (which contains URLs to the listings on Airbnb.com).
    destination_directory should be the path to the destination folder, where the results will be written into a csv file.
    city_name (optional) tells us the conventional suffix. If none is given, we will infer it from the listing.csv file.
    picture limit (optional) specifies a maximum number of images to extract keywords from in each listing. Each image takes ~3 seconds.
    confidence_threshold (optional) specifies the minimum confidence level extracted keywords must have.
    overwrite (optional) tells us whether to overwrite an existing file (of write a new one) or add to an existing one. 
    '''

    known_bad_urls = [
        'https://a0.muscache.com/pictures/adafb11b-41e9-49d3-908e-049dfd6934b6.jpg'
    ]

    # Read in the listings data.
    data = pd.read_csv(listing_directory)

    # Infer the city name if none is given.
    if city_name == None:
        city_name = listing_directory.split('_')[-1][:-4]

    # Check the file exists to be overwritten.
    if not os.path.exists('{}/image_keywords_{}.csv'.format(destination_directory, city_name)):
        # If the file doesn't exist, we take the same approach as if we were overwriting it.
        overwrite = True

    # Read in existing data keyword extraction data.
    if not overwrite:
        # Assume existing data is at './destination_directory/{city name}_image_keywords.csv'.
        existing_data = pd.read_csv('{}/image_keywords_{}.csv'.format(destination_directory, city_name)).reset_index()

    else:
        existing_data = pd.DataFrame(columns=['id', 'keywords', 'confindences'])
        existing_data.to_csv('{}/image_keywords_{}.csv'.format(destination_directory, city_name), index=False)

    # Get a list of all listing ids that do not have existing data.
    ids = [id for id in data.id.values if id not in existing_data.id.values]

    for id in ids:
        try:
            # Get the corresponding listing's url and primary photo.
            listing_url = data[data.id == id].listing_url.values[0]
            primary_photo = data[data.id == id].picture_url.values[0]

            # Get the image urls associated with the listing.
            image_urls = image_labels.get_listing_pictures(listing_url)

            # Remove any known bad URLs.
            for url in known_bad_urls:
                if url in image_urls:
                    image_urls.remove(url)

            # If no URLs remain, add in the primary photo.
            if len(image_urls) < 1:
                image_urls.append(primary_photo)

            # Restrict the number of images to be considered.
            n = min(len(image_urls), picture_limit)
            image_urls = random.sample(image_urls, n)

            # If the primary photo has not been selected, then replace the last choice with it.
            if primary_photo not in image_urls:
                image_urls = image_urls[:-1]
                image_urls.append(primary_photo)

            # Create the keyword dictionary...

            keyword_dict = {}

            for url in image_urls:
                labels = image_labels.get_labels(url)
                keywords = [(label['Name'], label['Confidence']) for label in labels['Labels'] if label['Confidence'] >= confidence_threshold]

                for keyword in keywords:
                    if keyword not in keyword_dict.keys():
                        keyword_dict[keyword[0]] = [keyword[1]]

                    else:
                        keyword_dict[keyword[0]].append(keyword[1])

            for keyword in keyword_dict.keys():
                keyword_dict[keyword] = np.mean(keyword_dict[keyword])

            # Incrementally add the new entry to the existing data.
            #new_entry = pd.DataFrame({'id' : id, 'keyword_dict' : keyword_dict})

            # EXPERIMENTAL:
            # Create new entry.
            new_entry = pd.DataFrame({'id' : [id], 'keywords' : [list(keyword_dict.keys())], 'confidences' : [list(keyword_dict.values())]})

            new_entry.to_csv('{}/image_keywords_{}.csv'.format(destination_directory, city_name), mode='a', header=False, index=False)

        except:
            # Add an empty entry (so that the index can be seen as considered if not overwriting).
            new_entry = pd.DataFrame({'id' : [id], 'keywords' : np.NaN, 'confidences' : np.NaN})

            new_entry.to_csv('{}/image_keywords_{}.csv'.format(destination_directory, city_name), mode='a', header=False, index=False)


# Function for computing the sentiment of listing images based on their extracted keywords.
def compute_image_sentiments(keyword_directory):
    '''
    This function will compute the mean (and standard deviation) sentiment for lists of keywords (and confidences).
    In each listing, each extracted keyword will be analysised for sentiment, and each sentiment will be scaled by the confidence score.
    The return will be a dataframe with columns 'id', 'mean_sentiment', and 'std_sentiment', and 'extracted_count'.
    
    keyword_directory should be a path to the image_keywords.csv containing keywords and sentiments for a list of ids.
    '''

    # Get the keyword data.
    data = pd.read_csv(keyword_directory).dropna()
    
    # New dataframe for resutls.
    results = pd.DataFrame(columns=['id', 'mean_sentiment', 'std_sentiment', 'extraction_count'])

    for i in range(len(data.index)):
        entry = data.iloc[i]

        keywords = ast.literal_eval(entry.keywords)
        confidences = ast.literal_eval(entry.confindences)

        # Compute the sentiment scores for the entry's extracted keywords.
        sentiments = []
        for j in range(len(keywords)):
            score = TextBlob(keywords[j]).sentiment.polarity * (confidences[j] / 100)
            sentiments.append(score)

        results.loc[i] = [entry.id, np.mean(sentiments), np.std(sentiments), len(sentiments)]

    return results
