# Concerning the RQs

The large majority of the exploration of our three RQs can be found in `testing.ipynb`. In there, we show any findings that have lead us to change our approach or look towards different areas.

In the section of `testing.ipynb` entitled *Miscellaneous Fun*, we walk through the data preprocessing step, create necessary features, and then go about answering RQ1. RQ2 and RQ3 are still in the works, and they are explored in `Predictive_Model_Examples.ipynb`.

## In Progress Developments
We are currently looking to clean up the running of this project by collecting all experiments into a single notebook -- `experiments.ipynb`. The runnning of the experiments themselves, and the handling of the data (e.g. preprocessing, file reading, etc.), is then to be abstracted into classes of the `RQs.py` script.

# What Are the Scripts For?

There are a variety of Python scripts where common methods are grouped together, usually in line with how they are discussed in the supporting literature so as to keep things nicely encapsulated and abstracted.

- `create_data.py` creates the master dataset CSV file. It does not handle data preprocessing -- this is done more experimentally at this stage, and so is conducted step-by-step in the `testing.ipynb` notebook (and was later given a method in `Regression_Model.py` for ease of use in building regression models).
- `image_keyword_features.py` uses the functionality of `image_labels.py` to extract keywords from listing images and save them to an image keywords CSV file. It is designed to *add* to a pre-existing CSV file, if one of the correct name is found, so that computation need only occur once -- there is significant overhead with generating these keywords, and so avoiding repetition is very much desired! This script also houses the function for computing the sentiment of images based on its keywords.
- `image_labels.py` handles interaction with AWS Rekognition and tackles the necessary web scraping for obtaining listing images from their given URL.
- `miscellaneous_helpers.py` will be a growing script to house functions that either don't have a well-defined group or that are widely used. In the latter case, this script will help in reducing dependencies by creating a common script for all others to rely on instead of creating an overly connected graph of dependency.
- `nlp_features.py` handles NLP tasks with unstructured data. In this script, we extract keywords from textual sources, create polarity and subjectivity features, and compute the perceived review sentiment.
- 'Regression_Model.py` is an evolving script handling the latter two RQs. In it, we will write methods for conducting regression, yeilding accuracy, and we may decide to retain a data preprocessing step as a method in this script instead of devoting another script to it.
- `success_metric.py` goes about the process of creating success scores for listings.

# What Are the Notebooks For?

The notebooks are very volatile files, but generally contain our working thoughts and experiments. They are constantly changing, being rewritten, or having their guts moved into dedicated scripts (thereby either leaving holes in them, or rendering them somewhat pointless though nonetheless a good relic of our research journey that might give an insight into what our thinking was at that time).
