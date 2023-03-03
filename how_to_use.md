# Using success_metric.py

To use success_metric.py, you must have [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), and [TextBlob](https://textblob.readthedocs.io/en/dev/) installed for Python.

You can then import this script into your own using `import success_metric` when in the same directory and call the `compute_scores` function to generate success scores for specified Airbnb listings.

`compute_scores` takes two strings denoting two distinct directories:
- `listing_directory` should be the directory path (as a string) to the listings.csv file downloaded from [Inside Airbnb](http://insideairbnb.com/get-the-data/) for whichever city you are interested in.
- `review_directory` is similar, except that it should specify the path to the reviews.csv file.

`compute_scores` also allows you to specify `review_rate`, which is used to estimate the number of bookings a listing has based on the number of reviews it receives. This value should be in the interval $(0,1]$, where $0$ indicates that no users leave reviews (which is obviously nonsensicle for listings with reviews), and $1$ indicates that every user leaves a review following their stay. We recommend $0.72$, but $0.5$ may represent a more appropriate choice.

# Using nlp_features.py

The same requirements persist for this script.

You can then import this script into your own using `import nlp_features` from within the same directory and call the `create_features` function to generate a dataframe of NLP features -- the sentiment polarity, sentiment subjectivity, and extracted keywords -- for an Airbnb listing's title, description, neighbourhood overview, and host about. The return will be a dataframe with corresponding IDs so that these features may be associated with other features generated elsewhere.

`create_features` takes three arguments:
- `listing_directory` should be the path (as a string) to the listings.csv files downloaded from [Inside Airbnb](http://insideairbnb.com/get-the-data/) for whichever city you are interested in.
- `keyword_limit` (optional) specifies the maximum number of keywords to be extracted from the description, neighbourhood overview, and host about. Be aware that there may be fewer (or no) keywords available to be extracted. Keywords will be returned in descending order of polarity.
- `title_keyword_limit` (optional) acts in the same way as `keyword_limit`, though only for the title.

# Using create_data.py

The primary function to call in this script is `city_data_generation`. It takes three arguments:
- `city` should be the string representation of the chosen city. It is expected to be in reference to file naming conventions -- files are expected to be named "listings_{city name}", where this city name is passed as the `city` argument.
- `datasets_directory` is the (string) path for the folder in which the listings.csv and review.csv files are kept. It is also the path to which the master.csv file will be written on completion.
- `latest_date` (optional) is the date (as a datetime object) used to determine recency of reviews -- recency will be the difference in days between the posted date and this given date (bounded to be positive). By default, this is set to today. 
