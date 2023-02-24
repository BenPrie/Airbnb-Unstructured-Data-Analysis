# Using success_metric.py

To use success_metric.py, you must have [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), and [TextBlob](https://textblob.readthedocs.io/en/dev/) installed for Python.

You can then import this script into your own using `import success_metric.py` and call the `compute_scores` function to generate success scores for specified Airbnb listings.

`compute_scores` takes two strings denoting two distinct directories:
- `listing_directory` should be the directory path (as a string) to the listings.csv file downloaded from [Inside Airbnb](http://insideairbnb.com/get-the-data/) for whichever city you are interested in.
- `review_directory` is similar, except that it should specify the path to the reviews.csv file.

`compute_scores` also allows you to specify `review_rate`, which is used to estimate the number of bookings a listing has based on the number of reviews it receives. This value should be in the interval $(0,1]$, where $0$ indicates that no users leave reviews (which is obviously nonsensicle for listings with reviews), and $1$ indicates that every user leaves a review following their stay. We recommend $0.72$, but $0.5$ may represent a more appropriate choice.
