# Imports as always...
import pandas as pd
from datetime import datetime
import numpy as np
import ast
from pathlib import Path

from scipy.stats import ttest_ind

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Outsourced functionality...
import success_metric as sm
import nlp_features as nlp
import miscellaneous_helpers as mh

# A data handling class that will manage the dataframes, preprocessing, etc.
class DataHandler():

    def __init__(self, city, datasets_directory, latest_date=datetime.now()):
        self.city = city
        self.datasets_directory = datasets_directory
        self.latest_date = latest_date
        self.master_data = pd.DataFrame()
        self.normalised_data = pd.DataFrame()


    # Generating the master dataset from scratch.
    def generate_master_data(self):
        # Establish the full paths to the raw data.
        listings_directory = self.datasets_directory + '/listings_' + self.city + '.csv'
        reviews_directory = self.datasets_directory + '/reviews_' + self.city + '.csv'

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
        success_scores = sm.compute_scores(listings_directory)
        nlp_features = nlp.create_features_revised(listings_directory, reviews_directory, latest_date=self.latest_date)

        # Merge the raw_data with the NLP features, retaining all listings.
        self.master_data = raw_data.merge(nlp_features, on='id', how='outer')

        # Read in the keyword data.
        keyword_data = pd.read_csv(self.datasets_directory + '/image_keywords_' + self.city + '.csv')
        keyword_data.rename({'keywords' : 'images_keywords', 'confindences' : 'images_confidences'}, axis=1, inplace=True)

        # Merge the master data with the keyword data.
        self.master_data = self.master_data.merge(keyword_data, on='id', how='outer')

        # Merge the master data with the success data.
        # We drop the first few columns because they are redundant indexing values.
        self.master_data = self.master_data.merge(success_scores, on='id', how='outer').reset_index().iloc[:,2:]

        # Write the master data to a new file.
        self.master_data.to_csv('{}/master_{}.csv'.format(self.datasets_directory, self.city))


    # Offloading mean keyword score calculation in images.
    def mean_keyword_scores(self, success_scores):
        # Define the keyword directory
        keyword_directory = '{}/image_keywords_{}.csv'.format(self.datasets_directory, self.city)

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


    # Helper function for converting dates to days since.
    def elapsed_days(self, from_date_as_string):
        from_date = datetime.strptime(from_date_as_string, '%Y-%m-%d')
        return max(0, (self.latest_date - from_date).days)


    # Generate the master data and preprocess.
    def prepare(self, overwrite=False, skewness_handling_steps=5):
        # Create master data for Edinburgh (if not created already).
        if not Path('{}/master_{}.csv'.format(self.datasets_directory, self.city)).is_file() or overwrite:
            self.generate_master_data()

        else:
            # Get the master data.
            self.master_data = pd.read_csv('{}/master_{}.csv'.format(self.datasets_directory, self.city))

        # Drop features used in success score definition.
        self.master_data.drop(columns=[
            'price',
            'minimum_nights_avg_ntm',
            'number_of_reviews_ltm', 
            'review_scores_rating'
        ], inplace=True, axis=1)

        # It would also be in the spirit of things to remove other review scores (e.g. for cleanliness), as that is sort of cheating.
        self.master_data.drop(columns=[
            'review_scores_accuracy', 
            'review_scores_cleanliness',
            'review_scores_checkin', 
            'review_scores_communication',
            'review_scores_location', 
            'review_scores_value'
        ], inplace=True, axis=1)

        # Drop unhelpful features.
        self.master_data.drop(columns=[
            'calculated_host_listings_count_shared_rooms',
            'neighbourhood',
            'neighbourhood_cleansed',
            'property_type',
            'bathrooms_text',
            'amenities',
        ], inplace=True, axis=1)

        # CONVERT TO NUMERICAL:

        # Convert pseudonumeric types (e.g. dates) to numeric...

        self.master_data.host_since = self.master_data.host_since.apply(self.elapsed_days)

        self.master_data.host_response_rate = self.master_data.host_response_rate.apply(
            lambda x : float(x[:-1]) if not pd.isna(x) 
            else x
        )

        self.master_data.host_acceptance_rate = self.master_data.host_acceptance_rate.apply(
            lambda x : float(x[:-1]) if not pd.isna(x) 
            else x
        )

        self.master_data.host_verifications = self.master_data.host_verifications.apply(
            lambda x : len(x)
        )

        self.master_data.first_review = self.master_data.first_review.apply(
            lambda x : self.elapsed_days(x) if not pd.isna(x)
            else x
        )

        self.master_data.last_review = self.master_data.last_review.apply(
            lambda x : self.elapsed_days(x) if not pd.isna(x)
            else x
        )

        # Convert categorical (e.g. boolean) types to numeric...

        self.master_data.host_response_time = self.master_data.host_response_time.map(
            lambda x : {'within an hour' : 1, 'within a few hours' : 2, 'within a day' : 3, 'a few days or more' : 4}.get(x, 0)
        )

        self.master_data.host_is_superhost = self.master_data.host_is_superhost.map(
            lambda x : {'t' : 1, 'f' : 0}.get(x, 0)
        )

        self.master_data.host_identity_verified = self.master_data.host_identity_verified.map(
            lambda x : {'t' : 1, 'f' : 0}.get(x, 0)
        )

        self.master_data.room_type = self.master_data.room_type.map(
            lambda x : {'Entire home/apt' : 1, 'Private room' : 2, 'Hotel room' : 3, 'Shared room' : 4}.get(x, 0)
        )

        self.master_data.instant_bookable = self.master_data.instant_bookable.map(
            lambda x : {'t' : 1, 'f' : 0}.get(x, 0)
        )

        self.master_data.host_has_profile_pic = self.master_data.host_has_profile_pic.map(
            lambda x : {'t' : 1, 'f' : 0}.get(x, 0)
        )

        self.master_data.has_availability = self.master_data.has_availability.map(
            lambda x : {'t' : 1, 'f' : 0}.get(x, 0)
        )

        # HANDLE MISSING VALUES:

        # Count the number of missing values in each feature.
        missing_values = self.master_data.isna().sum().sort_values(ascending=False)

        # Drop features with majority missing values.
        threshold = self.master_data.shape[0] // 2
        self.master_data.drop(columns=missing_values[missing_values > threshold].index, inplace=True, axis=1)

        # Drop rows with missing a succuess score.
        self.master_data = self.master_data[self.master_data.success_score.notna()]

        # Drop rows missing image label extractions.
        self.master_data = self.master_data[self.master_data.images_keywords.notna()]

        # Replace missing sentiment analyses with 0 scores.
        self.master_data.title_neg = self.master_data.title_neg.fillna(0)
        self.master_data.title_neu = self.master_data.title_neu.fillna(0)
        self.master_data.title_pos = self.master_data.title_pos.fillna(0)
        self.master_data.title_compound = self.master_data.title_compound.fillna(0)
        self.master_data.description_neg = self.master_data.description_neg.fillna(0)
        self.master_data.description_neu = self.master_data.description_neu.fillna(0)
        self.master_data.description_pos = self.master_data.description_pos.fillna(0)
        self.master_data.description_compound = self.master_data.description_compound.fillna(0)
        self.master_data.neighborhood_overview_neg = self.master_data.neighborhood_overview_neg.fillna(0)
        self.master_data.neighborhood_overview_neu = self.master_data.neighborhood_overview_neu.fillna(0)
        self.master_data.neighborhood_overview_pos = self.master_data.neighborhood_overview_pos.fillna(0)
        self.master_data.neighborhood_overview_compound = self.master_data.neighborhood_overview_compound.fillna(0)
        self.master_data.host_about_neg = self.master_data.host_about_neg.fillna(0)
        self.master_data.host_about_neu = self.master_data.host_about_neu.fillna(0)
        self.master_data.host_about_pos = self.master_data.host_about_pos.fillna(0)
        self.master_data.host_about_compound = self.master_data.host_about_compound.fillna(0)
        self.master_data.perceived_review_neg = self.master_data.perceived_review_neg.fillna(0)
        self.master_data.perceived_review_neu = self.master_data.perceived_review_neu.fillna(0)
        self.master_data.perceived_review_pos = self.master_data.perceived_review_pos.fillna(0)
        self.master_data.perceived_review_compound = self.master_data.perceived_review_compound.fillna(0)

        # Replace missing host_response_rate, host_acceptance_rate with mean values.
        self.master_data.host_response_rate = self.master_data.host_response_rate.fillna(self.master_data.host_response_rate.mean())
        self.master_data.host_acceptance_rate = self.master_data.host_acceptance_rate.fillna(self.master_data.host_acceptance_rate.mean()) 

        # Replace missing host_response_time with mode values.
        self.master_data.host_response_time = self.master_data.host_response_time.fillna(self.master_data.host_response_time.mode().values[0])

        # Replace missing bedroom and beds with mean values.
        self.master_data.bedrooms = self.master_data.bedrooms.fillna(self.master_data.bedrooms.mean())
        self.master_data.beds = self.master_data.beds.fillna(self.master_data.beds.mean())

        # Remove all remaining rows with missing values.
        self.master_data = self.master_data.dropna()

        # CREATE NEW FEATURES:

        # Calcualte the success of the image keywords.
        keyword_success = self.mean_keyword_scores(self.master_data[['id', 'success_score']])

        # For each listing, compute the mean and standard deviation in the perceived success of image keywords.
        weighted_image_score_mean = []
        weighted_image_score_max = []
        weighted_image_score_min = []
        weighted_image_score_std = []

        for i in range(len(self.master_data.index)):
            entry = self.master_data.iloc[i]

            keywords = ast.literal_eval(entry.images_keywords)
            confidences = ast.literal_eval(entry.images_confidences)

            scores = []
            for j in range(len(keywords)):
                scores.append(keyword_success[keywords[j]] * (confidences[j] / 100)) 

            weighted_image_score_mean.append(np.mean(scores))
            weighted_image_score_max.append(max(scores))
            weighted_image_score_min.append(min(scores))
            weighted_image_score_std.append(np.std(scores))

        self.master_data['weighted_image_score_mean'] = weighted_image_score_mean
        self.master_data['weighted_image_score_max'] = weighted_image_score_max
        self.master_data['weighted_image_score_min'] = weighted_image_score_min
        self.master_data['weighted_image_score_std'] = weighted_image_score_std

        # HANDLE SKEWNESS:
        for i in range(skewness_handling_steps):
            # Calculate the skewness of the features.
            skewness = self.master_data.skew().sort_values(ascending=False)

            # Deal with positive skewness by performing a square root transformation.
            for skew_feature in skewness[skewness > 1].index:
                self.master_data[skew_feature] = np.power(self.master_data[skew_feature], 1/2)

            # Deal with negative skewness by performing a square transformation.
            for skew_feature in skewness[skewness < -1].index:
                self.master_data[skew_feature] = np.power(self.master_data[skew_feature], 2)

        # NORMALISATION:

        # Create the scaler object
        scaler = StandardScaler()

        # Normalise the data
        data_to_normalise = self.master_data.drop(columns=['images_keywords', 'images_confidences'], axis=1)
        self.normalised_data = pd.DataFrame(scaler.fit_transform(data_to_normalise), columns=data_to_normalise.columns)

        # Remove introduded missing values (?).
        self.normalised_data = self.normalised_data.dropna()

        return self.normalised_data


class RQ1():

    def __init__(self, normalised_data, threshold=0.5):
        # We take the bottom and top {threshold}% when considered unsuccessful and successful listings.
        threshold = min(abs(threshold), 0.5)
        self.unsuccessful_data = normalised_data[normalised_data.success_score < normalised_data.success_score.quantile(threshold)]
        self.successful_data = normalised_data[normalised_data.success_score > normalised_data.success_score.quantile(1 - threshold)]


    def get_data_split(self):
        return self.unsuccessful_data, self.successful_data


    # Main method for running the experiment.
    def run (self, features_to_consider, sig_level=0.01):
        results = pd.DataFrame(columns=[
            'Feature', 
            'No. Unsuccessful', 
            'No. Successful', 
            'Unsuccessful Mean', 
            'Successful Mean',
            'Unsuccessful Std', 
            'Successful Std',
            'Unsuccessful Skew', 
            'Successful Skew',
            'p-value', 
            'Rejection Decision'
        ])

        for feature in features_to_consider:
            less_sample = self.unsuccessful_data[feature]
            more_sample = self.successful_data[feature]

            _, double_p = ttest_ind(more_sample, less_sample, equal_var=False)

            if np.mean(more_sample) > np.mean(less_sample):
                p_value = double_p / 2

            else:
                p_value = 1 - (double_p / 2)

            # Append the results.
            feature_results = pd.DataFrame({
                'Feature' : [feature], 
                'No. Unsuccessful' : [len(less_sample)], 
                'No. Successful' : [len(more_sample)], 
                'Unsuccessful Mean' : [np.mean(less_sample)], 
                'Successful Mean' : [np.mean(more_sample)],
                'Unsuccessful Std' : [np.std(less_sample)], 
                'Successful Std' : [np.std(more_sample)],
                'Unsuccessful Skew' : [less_sample.skew()], 
                'Successful Skew' : [more_sample.skew()],
                'p-value' : [p_value], 
                'Rejection Decision' : [p_value < sig_level]
            })

            results = pd.concat([results, feature_results])

        return results

            


# Class for conducting RQ3.
class RQ3():

    def __init__(self, normalised_data, structured_features, unstructured_features, hybrid_features, pca=True, threshold_explained_variance=0.9):
        self.normalised_data = normalised_data
        
        # Define the input datasets.
        self.X_struct = self.normalised_data[structured_features]
        self.X_unstruct = self.normalised_data[unstructured_features]
        self.X_hybrid = self.normalised_data[hybrid_features]

        # Define the target variable.
        self.y = self.normalised_data.success_score

        if pca:
            # Reduce the dimensionality as much as possible while retaining some threshold variance explanation.

            struct_pca = PCA()
            self.X_struct = struct_pca.fit_transform(X=self.X_struct, y=self.y)

            for i in range(len(struct_pca.components_)):
                if sum(struct_pca.explained_variance_ratio_[:i]) > threshold_explained_variance:
                    break

            self.X_struct = pd.DataFrame(self.X_struct).iloc[:,:i]

            unstruct_pca = PCA()
            self.X_unstruct = unstruct_pca.fit_transform(X=self.X_unstruct, y=self.y)


            for i in range(len(unstruct_pca.components_)):
                if sum(unstruct_pca.explained_variance_ratio_[:i]) > threshold_explained_variance:
                    break

            self.X_unstruct = pd.DataFrame(self.X_unstruct).iloc[:,:i]

            hybrid_pca = PCA()
            self.X_hybrid = hybrid_pca.fit_transform(X=self.X_hybrid, y=self.y)

            for i in range(len(hybrid_pca.components_)):
                if sum(hybrid_pca.explained_variance_ratio_[:i]) > threshold_explained_variance:
                    break

            self.X_hybrid = pd.DataFrame(self.X_hybrid).iloc[:,:i]


    # Main method for running the experiment.
    def run(self,
        linear_regression=True,
        support_vector_regression=True,
        multilayer_perceptron=True,
        gaussian_process_regression=True,
        decision_tree_regression=True,
        random_forest_regression=True,
        seed=1
    ):
        results = pd.DataFrame(columns=['Model', 'Dataset', 'MAE', 'MSE', 'R2 Score'])

        # Train-test split for each dataset.
        X_struct_train, X_struct_test, y_struct_train, y_struct_test = train_test_split(self.X_struct, self.y, test_size=0.2, random_state=seed)
        X_unstruct_train, X_unstruct_test, y_unstruct_train, y_unstruct_test = train_test_split(self.X_unstruct, self.y, test_size=0.2, random_state=seed)
        X_hybrid_train, X_hybrid_test, y_hybrid_train, y_hybrid_test = train_test_split(self.X_hybrid, self.y, test_size=0.2, random_state=seed)

        if linear_regression:
            lin_reg_struct = LinearRegression().fit(X_struct_train, y_struct_train)
            y_struct_pred = lin_reg_struct.predict(X_struct_test)
            struct_mae = mean_squared_error(y_true=y_struct_test, y_pred=y_struct_pred) 
            struct_mse = mean_absolute_error(y_true=y_struct_test, y_pred=y_struct_pred)
            struct_r2 = r2_score(y_true=y_struct_test, y_pred=y_struct_pred)

            lin_reg_unstruct = LinearRegression().fit(X_unstruct_train, y_unstruct_train)
            y_unstruct_pred = lin_reg_unstruct.predict(X_unstruct_test)
            unstruct_mae = mean_squared_error(y_true=y_unstruct_test, y_pred=y_unstruct_pred) 
            unstruct_mse = mean_absolute_error(y_true=y_unstruct_test, y_pred=y_unstruct_pred)
            unstruct_r2 = r2_score(y_true=y_unstruct_test, y_pred=y_unstruct_pred)

            lin_reg_hybrid = LinearRegression().fit(X_hybrid_train, y_hybrid_train)
            y_hybrid_pred = lin_reg_hybrid.predict(X_hybrid_test)
            hybrid_mae = mean_squared_error(y_true=y_hybrid_test, y_pred=y_hybrid_pred) 
            hybrid_mse = mean_absolute_error(y_true=y_struct_test, y_pred=y_hybrid_pred)
            hybrid_r2 = r2_score(y_true=y_hybrid_test, y_pred=y_hybrid_pred)

            # Append the results.
            lin_reg_results = pd.DataFrame({
                'Model' : ['Linear Regression'] * 3,
                'Dataset' : ['Structured', 'Unstructured', 'Hyrbid'],
                'MAE' : [struct_mae, unstruct_mae, hybrid_mae],
                'MSE' : [struct_mse, unstruct_mse, hybrid_mse],
                'R2 Score' : [struct_r2, unstruct_r2, hybrid_r2],
            })

            results = pd.concat([results, lin_reg_results])

        if support_vector_regression:
            svr_struct = SVR().fit(X_struct_train, y_struct_train)
            y_struct_pred = svr_struct.predict(X_struct_test)
            struct_mae = mean_squared_error(y_true=y_struct_test, y_pred=y_struct_pred) 
            struct_mse = mean_absolute_error(y_true=y_struct_test, y_pred=y_struct_pred)
            struct_r2 = r2_score(y_true=y_struct_test, y_pred=y_struct_pred)

            svr_unstruct = SVR().fit(X_unstruct_train, y_unstruct_train)
            y_unstruct_pred = svr_unstruct.predict(X_unstruct_test)
            unstruct_mae = mean_squared_error(y_true=y_unstruct_test, y_pred=y_unstruct_pred) 
            unstruct_mse = mean_absolute_error(y_true=y_unstruct_test, y_pred=y_unstruct_pred)
            unstruct_r2 = r2_score(y_true=y_unstruct_test, y_pred=y_unstruct_pred)

            svr_hybrid = SVR().fit(X_hybrid_train, y_hybrid_train)
            y_hybrid_pred = svr_hybrid.predict(X_hybrid_test)
            hybrid_mae = mean_squared_error(y_true=y_hybrid_test, y_pred=y_hybrid_pred) 
            hybrid_mse = mean_absolute_error(y_true=y_struct_test, y_pred=y_hybrid_pred)
            hybrid_r2 = r2_score(y_true=y_hybrid_test, y_pred=y_hybrid_pred)

            # Append the results.
            svr_results = pd.DataFrame({
                'Model' : ['Support Vector Regression (SVR)'] * 3,
                'Dataset' : ['Structured', 'Unstructured', 'Hyrbid'],
                'MAE' : [struct_mae, unstruct_mae, hybrid_mae],
                'MSE' : [struct_mse, unstruct_mse, hybrid_mse],
                'R2 Score' : [struct_r2, unstruct_r2, hybrid_r2],
            })

            results = pd.concat([results, svr_results])

        if multilayer_perceptron:
            mlp_struct = MLPRegressor().fit(X_struct_train, y_struct_train)
            y_struct_pred = mlp_struct.predict(X_struct_test)
            struct_mae = mean_squared_error(y_true=y_struct_test, y_pred=y_struct_pred) 
            struct_mse = mean_absolute_error(y_true=y_struct_test, y_pred=y_struct_pred)
            struct_r2 = r2_score(y_true=y_struct_test, y_pred=y_struct_pred)

            mlp_unstruct = MLPRegressor().fit(X_unstruct_train, y_unstruct_train)
            y_unstruct_pred = mlp_unstruct.predict(X_unstruct_test)
            unstruct_mae = mean_squared_error(y_true=y_unstruct_test, y_pred=y_unstruct_pred) 
            unstruct_mse = mean_absolute_error(y_true=y_unstruct_test, y_pred=y_unstruct_pred)
            unstruct_r2 = r2_score(y_true=y_unstruct_test, y_pred=y_unstruct_pred)

            mlp_hybrid = MLPRegressor().fit(X_hybrid_train, y_hybrid_train)
            y_hybrid_pred = mlp_hybrid.predict(X_hybrid_test)
            hybrid_mae = mean_squared_error(y_true=y_hybrid_test, y_pred=y_hybrid_pred) 
            hybrid_mse = mean_absolute_error(y_true=y_struct_test, y_pred=y_hybrid_pred)
            hybrid_r2 = r2_score(y_true=y_hybrid_test, y_pred=y_hybrid_pred)

            # Append the results.
            mlp_results = pd.DataFrame({
                'Model' : ['Multi-Layer Perceptron (MLP)'] * 3,
                'Dataset' : ['Structured', 'Unstructured', 'Hyrbid'],
                'MAE' : [struct_mae, unstruct_mae, hybrid_mae],
                'MSE' : [struct_mse, unstruct_mse, hybrid_mse],
                'R2 Score' : [struct_r2, unstruct_r2, hybrid_r2],
            })

            results = pd.concat([results, mlp_results])

        if gaussian_process_regression:
            gauss_struct = GaussianProcessRegressor().fit(X_struct_train, y_struct_train)
            y_struct_pred = gauss_struct.predict(X_struct_test)
            struct_mae = mean_squared_error(y_true=y_struct_test, y_pred=y_struct_pred) 
            struct_mse = mean_absolute_error(y_true=y_struct_test, y_pred=y_struct_pred)
            struct_r2 = r2_score(y_true=y_struct_test, y_pred=y_struct_pred)

            gauss_unstruct = GaussianProcessRegressor().fit(X_unstruct_train, y_unstruct_train)
            y_unstruct_pred = gauss_unstruct.predict(X_unstruct_test)
            unstruct_mae = mean_squared_error(y_true=y_unstruct_test, y_pred=y_unstruct_pred) 
            unstruct_mse = mean_absolute_error(y_true=y_unstruct_test, y_pred=y_unstruct_pred)
            unstruct_r2 = r2_score(y_true=y_unstruct_test, y_pred=y_unstruct_pred)

            gauss_hybrid = GaussianProcessRegressor().fit(X_hybrid_train, y_hybrid_train)
            y_hybrid_pred = gauss_hybrid.predict(X_hybrid_test)
            hybrid_mae = mean_squared_error(y_true=y_hybrid_test, y_pred=y_hybrid_pred) 
            hybrid_mse = mean_absolute_error(y_true=y_struct_test, y_pred=y_hybrid_pred)
            hybrid_r2 = r2_score(y_true=y_hybrid_test, y_pred=y_hybrid_pred)

            # Append the results.
            gauss_results = pd.DataFrame({
                'Model' : ['Gaussian Process Regression'] * 3,
                'Dataset' : ['Structured', 'Unstructured', 'Hyrbid'],
                'MAE' : [struct_mae, unstruct_mae, hybrid_mae],
                'MSE' : [struct_mse, unstruct_mse, hybrid_mse],
                'R2 Score' : [struct_r2, unstruct_r2, hybrid_r2],
            })

            results = pd.concat([results, gauss_results])

        if decision_tree_regression:
            dt_struct = DecisionTreeRegressor().fit(X_struct_train, y_struct_train)
            y_struct_pred = dt_struct.predict(X_struct_test)
            struct_mae = mean_squared_error(y_true=y_struct_test, y_pred=y_struct_pred) 
            struct_mse = mean_absolute_error(y_true=y_struct_test, y_pred=y_struct_pred)
            struct_r2 = r2_score(y_true=y_struct_test, y_pred=y_struct_pred)

            dt_unstruct = DecisionTreeRegressor().fit(X_unstruct_train, y_unstruct_train)
            y_unstruct_pred = dt_unstruct.predict(X_unstruct_test)
            unstruct_mae = mean_squared_error(y_true=y_unstruct_test, y_pred=y_unstruct_pred) 
            unstruct_mse = mean_absolute_error(y_true=y_unstruct_test, y_pred=y_unstruct_pred)
            unstruct_r2 = r2_score(y_true=y_unstruct_test, y_pred=y_unstruct_pred)

            dt_hybrid = DecisionTreeRegressor().fit(X_hybrid_train, y_hybrid_train)
            y_hybrid_pred = dt_hybrid.predict(X_hybrid_test)
            hybrid_mae = mean_squared_error(y_true=y_hybrid_test, y_pred=y_hybrid_pred) 
            hybrid_mse = mean_absolute_error(y_true=y_struct_test, y_pred=y_hybrid_pred)
            hybrid_r2 = r2_score(y_true=y_hybrid_test, y_pred=y_hybrid_pred)

            # Append the results.
            dt_results = pd.DataFrame({
                'Model' : ['Decision Tree Regression'] * 3,
                'Dataset' : ['Structured', 'Unstructured', 'Hyrbid'],
                'MAE' : [struct_mae, unstruct_mae, hybrid_mae],
                'MSE' : [struct_mse, unstruct_mse, hybrid_mse],
                'R2 Score' : [struct_r2, unstruct_r2, hybrid_r2],
            })

            results = pd.concat([results, dt_results])

        if random_forest_regression:
            rf_struct = RandomForestRegressor().fit(X_struct_train, y_struct_train)
            y_struct_pred = rf_struct.predict(X_struct_test)
            struct_mae = mean_squared_error(y_true=y_struct_test, y_pred=y_struct_pred) 
            struct_mse = mean_absolute_error(y_true=y_struct_test, y_pred=y_struct_pred)
            struct_r2 = r2_score(y_true=y_struct_test, y_pred=y_struct_pred)

            rf_unstruct = RandomForestRegressor().fit(X_unstruct_train, y_unstruct_train)
            y_unstruct_pred = rf_unstruct.predict(X_unstruct_test)
            unstruct_mae = mean_squared_error(y_true=y_unstruct_test, y_pred=y_unstruct_pred) 
            unstruct_mse = mean_absolute_error(y_true=y_unstruct_test, y_pred=y_unstruct_pred)
            unstruct_r2 = r2_score(y_true=y_unstruct_test, y_pred=y_unstruct_pred)

            rf_hybrid = RandomForestRegressor().fit(X_hybrid_train, y_hybrid_train)
            y_hybrid_pred = rf_hybrid.predict(X_hybrid_test)
            hybrid_mae = mean_squared_error(y_true=y_hybrid_test, y_pred=y_hybrid_pred) 
            hybrid_mse = mean_absolute_error(y_true=y_struct_test, y_pred=y_hybrid_pred)
            hybrid_r2 = r2_score(y_true=y_hybrid_test, y_pred=y_hybrid_pred)

            # Append the results.
            rf_results = pd.DataFrame({
                'Model' : ['Random Forest Regression'] * 3,
                'Dataset' : ['Structured', 'Unstructured', 'Hyrbid'],
                'MAE' : [struct_mae, unstruct_mae, hybrid_mae],
                'MSE' : [struct_mse, unstruct_mse, hybrid_mse],
                'R2 Score' : [struct_r2, unstruct_r2, hybrid_r2],
            })

            results = pd.concat([results, rf_results])

        return results