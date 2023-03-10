import pandas as pd
from datetime import datetime
from pathlib import Path
import seaborn as sns
import numpy as np
from scipy.stats import skew
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import linear_model
import create_data
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
import ast

# Helper function for converting dates to days since. Directly copied from model_comparisons.ipynb
def elapsed_days(from_date_as_string, to_date=datetime(2022, 12, 16)):
    from_date = datetime.strptime(from_date_as_string, '%Y-%m-%d')
    return max(0, (to_date - from_date).days)


def pre_process(city):
    '''
    This function encompasses all of the pre-processing steps that Ben carried out in model_comparisons.ipynb. 
    The data from each city should require the same pre-processing steps. 
    Most of this code is directecly ripped from model_comparisons.ipynb.
    The function accepts the name of the city a string and returns the complete processed dataset for the city. 
    '''
    
    # Create master data if not already created.
    if not Path('datasets/master_{}.csv'.format(city)).is_file():
        create_data.city_data_generation(city, 'datasets', datetime(2022, 12, 16))
        print('Master data file generated.')
    else:
        print('Pre-existing master data file found (no new data created).')
       
    
    # Get the data.
    master_data = pd.read_csv('datasets/master_{}.csv'.format(city)).iloc[:,2:]
        
        
    # We will be predicting success so, it would only be fair for us to remove all data used to generate success scores.
    master_data.drop(columns=[
        'price', 
        'minimum_nights_avg_ntm', 
        'number_of_reviews_ltm', 
        'review_scores_rating'
    ], inplace=True, axis=1)

    # It would also be in the spirit of things to remove other review scores (e.g. for cleanliness), as that is sort of cheating.
    master_data.drop(columns=[
        'review_scores_accuracy', 
        'review_scores_cleanliness',
        'review_scores_checkin', 
        'review_scores_communication',
        'review_scores_location', 
        'review_scores_value'
    ], inplace=True, axis=1)
    
    
    # Get the missing value counts and drop features with majority missing values.
    missing_values = master_data.isna().sum()
    threshold = master_data.shape[0] // 2
    master_data.drop(columns=missing_values[missing_values > threshold].index, inplace=True, axis=1)
    
    
    # Drop rows with missing a succuess score.
    master_data = master_data[master_data.success_score.notna()]
    
    
    # Remove unhelpful features (for prediction of success).
    master_data.drop(columns=[
        'host_id',
        'neighbourhood',
        'neighbourhood_cleansed',
        'latitude',
        'longitude',
        'property_type',
        'bathrooms_text',
        'amenities',
        'minimum_minimum_nights', 
        'maximum_minimum_nights',
        'minimum_maximum_nights', 
        'maximum_maximum_nights',
        'maximum_nights_avg_ntm',
        'calculated_host_listings_count',
        'calculated_host_listings_count_entire_homes',
        'calculated_host_listings_count_private_rooms',
        'calculated_host_listings_count_shared_rooms',
        'title_keywords',
        'description_keywords', 
        'neighborhood_overview_keywords',
        'host_about_keywords'
    ], inplace=True, axis=1) 

    
    # Remove features that almost certinaly don't add anything (e.g. almost all the same value).
    master_data.drop(columns=[
        'has_availability',
        'host_has_profile_pic'
    ], inplace=True, axis=1)

    
    # Convert pseudonumeric types (e.g. dates) to numeric...

    master_data.host_since = master_data.host_since.apply(elapsed_days)

    master_data.host_response_rate = master_data.host_response_rate.apply(
        lambda x : float(x[:-1]) if not pd.isna(x) 
        else x
    )

    master_data.host_acceptance_rate = master_data.host_acceptance_rate.apply(
        lambda x : float(x[:-1]) if not pd.isna(x) 
        else x
    )

    master_data.host_verifications = master_data.host_verifications.apply(
        lambda x : len(x)
    )

    master_data.first_review = master_data.first_review.apply(
        lambda x : elapsed_days(x) if not pd.isna(x)
        else x
    )

    master_data.last_review = master_data.last_review.apply(
        lambda x : elapsed_days(x) if not pd.isna(x)
        else x
    )
    
    
    # Replace missing polarity and subjectivity scores with 0 (i.e. neutral and factual).
    master_data.host_about_polarity = master_data.host_about_polarity.fillna(0)
    master_data.host_about_subjectivity = master_data.host_about_subjectivity.fillna(0)
    master_data.neighborhood_overview_polarity = master_data.neighborhood_overview_polarity.fillna(0)
    master_data.neighborhood_overview_subjectivity = master_data.neighborhood_overview_subjectivity.fillna(0)
    master_data.description_polarity = master_data.description_polarity.fillna(0)
    master_data.description_subjectivity = master_data.description_subjectivity.fillna(0)

    # Replace missing host_response_rate, host_acceptance_rate with mean values.
    master_data.host_response_rate = master_data.host_response_rate.fillna(master_data.host_response_rate.mean())
    master_data.host_acceptance_rate = master_data.host_acceptance_rate.fillna(master_data.host_acceptance_rate.mean()) 

    # Replace missing host_response_time with mode values.
    master_data.host_response_time = master_data.host_response_time.fillna(master_data.host_response_time.mode().values[0])
        
        
    # Remove all rows with missing values still remaining.
    master_data = master_data.dropna()
    
    
    # Calculate the skewness of the features.
    skewness = master_data.skew().sort_values(ascending=False)
    skewness[abs(skewness) > 1]

    
    # Don't worry about skewness with the success scores.
    skewness.success_score = 0
    
    
    # Deal with positive skewness by performing a square root transformation.
    for skew_feature in skewness[skewness > 1.5].index:
        master_data[skew_feature] = np.power(master_data[skew_feature], 1/2)
        

    # Deal with negative skewness by performing a square transformation.
    for skew_feature in skewness[skewness < -1.5].index:
        master_data[skew_feature] = np.power(master_data[skew_feature], 2)
        
    
    # Re-assess skewness.
    skewness = master_data.skew().sort_values(ascending=False)
    skewness[abs(skewness) > 1]
    

    # Don't worry about skewness with the success scores.
    skewness.success_score = 0
    
    
    # Deal with positive skewness AGAIN by performing a square root transformation AGAIN (i.e. fourth root).
    for skew_feature in skewness[skewness > 1.5].index:
        master_data[skew_feature] = np.power(master_data[skew_feature], 1/2)
        

    # Deal with negative skewness AGAIN by performing a square transformation AGAIN (i.e. to power 4).
    for skew_feature in skewness[skewness < -1.5].index:
        master_data[skew_feature] = np.power(master_data[skew_feature], 2)
        
        
    # Re-assess skewness.
    skewness = master_data.skew().sort_values(ascending=False)
    skewness[abs(skewness) > 1]
    

    # Don't worry about skewness with the success scores.
    skewness.success_score = 0


    # Calcualte the success of the image keywords.
    keyword_success = create_data.mean_keyword_scores(master_data[['id', 'success_score']], 'datasets/image_keywords_edinburgh.csv')
    
    # For each listing, compute the mean and standard deviation in the perceived success of image keywords.
    weighted_image_score_mean = []
    weighted_image_score_max = []
    weighted_image_score_min = []
    weighted_image_score_std = []

    for i in range(len(master_data.index)):
        entry = master_data.iloc[i]

        keywords = ast.literal_eval(entry.images_keywords)
        confidences = ast.literal_eval(entry.images_confidences)

        scores = []
        for j in range(len(keywords)):
            scores.append(keyword_success[keywords[j]] * (confidences[j] / 100)) 

        weighted_image_score_mean.append(np.mean(scores))
        weighted_image_score_max.append(max(scores))
        weighted_image_score_min.append(min(scores))
        weighted_image_score_std.append(np.std(scores))

    master_data['weighted_image_score_mean'] = weighted_image_score_mean
    master_data['weighted_image_score_max'] = weighted_image_score_max
    master_data['weighted_image_score_min'] = weighted_image_score_min
    master_data['weighted_image_score_std'] = weighted_image_score_std

    # Drop image keywords and confidences, now that we have used them.
    master_data = master_data.drop(columns=['images_keywords', 'images_confidences'], axis=1)

    # Convert categorical (e.g. boolean) types to numeric...

    master_data.host_response_time = master_data.host_response_time.map(
        lambda x : {'within an hour' : 1, 'within a few hours' : 2, 'within a day' : 3, 'a few days or more' : 4}.get(x, 0)
    )

    master_data.host_is_superhost = master_data.host_is_superhost.map(
        lambda x : {'t' : 1, 'f' : 0}.get(x, 0)
    )

    master_data.host_identity_verified = master_data.host_identity_verified.map(
        lambda x : {'t' : 1, 'f' : 0}.get(x, 0)
    )

    master_data.room_type = master_data.room_type.map(
        lambda x : {'Entire home/apt' : 1, 'Private room' : 2, 'Hotel room' : 3, 'Shared room' : 4}.get(x, 0)
    )

    master_data.instant_bookable = master_data.instant_bookable.map(
        lambda x : {'t' : 1, 'f' : 0}.get(x, 0)
    )
    
    # Return the processed data.
    return master_data



def create_subsets(data, raw_data=True, text_data=True):
    '''
    This function creates subsets of the main dataset.
    The function accepts a dataframe. 
    By default, two subsets are returned: raw data only and text data only.
    If only one of the subsets is needed, the other can be set to False.
    '''
    
    if raw_data == True:
        # Given structured data.
        raw_data = data[[
            'host_since', 
            'host_response_time', 
            'host_response_rate',
            'host_acceptance_rate', 
            'host_is_superhost', 
            'host_listings_count',
            'host_total_listings_count', 
            'host_verifications',
            'host_identity_verified', 
            'room_type', 
            'accommodates', 
            'bedrooms',
            'beds', 
            'minimum_nights', 
            'maximum_nights', 
            'availability_30',
            'availability_60', 
            'availability_90', 
            'availability_365',
            'number_of_reviews', 
            'number_of_reviews_l30d', 
            'first_review',
            'last_review', 
            'instant_bookable', 
            'reviews_per_month',
        ]]
        
    if text_data == True:
        # Inferred features from unstructured data.
        text_data = data[[
            'title_polarity', 
            #'title_subjectivity', 
            'description_polarity',
            #'description_subjectivity', 
            'neighborhood_overview_polarity',
            #'neighborhood_overview_subjectivity', 
            'host_about_polarity',
            #'host_about_subjectivity', 
            'perceived_review_sentiment',
            'min_review_sentiment', 
            'max_review_sentiment', 
            'weighted_image_score_mean', 
            'weighted_image_score_max',
            'weighted_image_score_min', 
            'weighted_image_score_std'
        ]]
        
    return raw_data, text_data


def predictive_model(data, masterdata, n=10):
    '''
    This function is a predictive model that can be applied to any dataset.
    Inputs:
    data - the data we want to train the model with (e.g. raw_data, text_data).
    masterdata -  the masterdataset for the city. This is needed to have access to the actual sucess scores.
    n - the number of fold for the kfold cross validation. The default is 10.
    The function returns 3 lists: MSE, MAE, and correlation between the predicted and actual success.
    The lists contain the calcualted value for each fold of the cross validation. 
    The function also prints the of each calculated value accross the folds.
    '''
    # Drop the success score if it is still included in the training data.
    if 'success_score' in data:
        data = data.drop('success_score', axis=1)
    
    # Set up lists.
    RMSE = []
    MAE = []
    coeff = []
    
    #set up cross validation.
    kf = KFold(n_splits=n, shuffle=True)
    
    # Split test and training sets. Loop over all folds.
    for train_index, test_index in kf.split(data):
        x_train, x_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = masterdata.iloc[train_index]['success_score'], masterdata.iloc[test_index]['success_score']
        
        # Initiate linear regression model.
        model = linear_model.LinearRegression()
        
        # Fit model.
        model.fit(x_train, y_train)
        
        # Make predictions.
        y_pred = model.predict(x_test)
        
        # Save results to lists.
        RMSE.append(np.sqrt(mean_squared_error(y_true = y_test, y_pred = y_pred)))
        MAE.append(mean_absolute_error(y_true = y_test, y_pred = y_pred))
        coeff.append(np.corrcoef(y_pred, y_test)[1][0])
        
    print ('The average root mean squarred error is ', np.mean(RMSE))
    print ('The average mean absolute error is ', np.mean(MAE))
    print ('The correlation between the predicted and actual success: is ', np.mean(coeff))
    return RMSE, MAE, coeff
