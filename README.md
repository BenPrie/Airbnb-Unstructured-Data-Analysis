# Problem and Motivation
Airbnb has become the top platform for short term property rentals worldwide, and has an enormous influence on real estate markets. (Garcia-Lopez et al., 2020; Barron and Kung, 2020; Koster et al., 2021; Sheppard and Udell, 2016)

Predicting the success of an Airbnb listing is crucial to *mitigate risk* when entering the market with new properties, and to *maximise profits* in the long-term:
- **Mitigating risk**: when entering the market with a new property, hosts will tend to experiment with pricing strategies, perhaps never finding the optimum. Predicting success ahead of time can give foresight into the optimal pricing strategy, allowing hosts to reduce experimentation time.
- **Maximising profit**: not knowing the success potential or appropriate pricing strategy for a property could reduce profits, either by pricing too low and reducing profit margins, pricing too high and losing business, or by developing properties in lower potential areas (e.g. investing in central heating may be wasted overhead if it does not increase the likelihood or degree of success enough).

Relying on structured data for this prediction is unduly trusting of data collectors and maintainers -- will they be accurate, will they be precise, will they be consistent? It is potential perilous to trust long-term predictive capability in the continuing accuracy and availability of structured data. (Ge and Harfield, 2007; Moreno, 2017; Battaglia et al., 2010)

Airbnb listings contain an abundance of unstructured data with huge potential for machine learning models -- listing images, property descriptions, user reviews, etc.

This all motivates the following question: **can unstructured data be exploited so as to preclude the necessity for accurate, manually collated, structured data without the loss of predictive performance**? If it is not enough to eliminate our dependence on structured data, then perhaps it can be used to improve existing models.

---

# Related Work
This paper extends the ideas of Kalehbasti et al. (2021) to draw inference about the *success* of Airbnb listings from their unstructured data rather than simply predict the price -- cheaper properties on a quicker turnaround that still maintain positive reviews may well be more successful than an expensive but seldom rented property.

---

# Methodology

## Defining the Success Metric
We should be very precise about the metrics we are using to quantify the success of Airbnb listings, and hence what we are making predictions about, and which aspects of a listing are under the microscope.

There are three main candidates for success:
- **Gross income** -- the most commonly adopted by researchers (Poursaeed et al., 2018; Kalehbasti et al., 2021) and perhaps the most easily inferred. In this case, success follows very closely from prediction of listing price (Kalehbasti et al., 2021) and inferred tenancy lengths. We might not wish to use gross income as our metric for success as it neglects the renters' experiences, which are certain to have a major influence in the long-term 'success' of a listing.
- **Review sentiment** -- a less monetarily-oriented philosophy determines the success of a listing more philanthropically by taking the renters' written experiences (i.e. their public reviews) as a proxy for the quality of their tenancy (i.e. success) (Lawani et al., 2019). This can also act as a replacement for Airbnb's property rating system, which could be seem as too arbitrary a metric (for instance, two people may have different thresholds for a perfect rating). While review sentiment is heavily biased as it is based entirely on a collection of individuals' opinions (Kiritchenko and Mohammad, 2018), it is certainly not arbitrary, as it reflects the written opinions of everyone on the same scale.
- **Booking Rate** -- to maintain a business-oriented mindset while avoiding the notion of *cash is king*, you might judge a property to be more successful if it has a high booking rate. That is, success is defined as the proportion of a given time period it spends rented out. While this is an easily inferred metric, it deprives us of a sense of value -- if a property is rented to a tenant that hates their stay and leaves after a week, it should not be equally as successful as another property is rented out to the same tenant for one day periods seven days in a row.

To exercise a flexible and all-encompassing philosophy, we will define success through a blend of these candidates.

Hosts are predominantly driven by revenue potential, so their listing price is of principle importance. We would then like to incorporate the sentiment of reviews to remain vigilant to changes in the value listing or the likelihood that a booking is made. Studies have also found that user reviews are hugely influential in consumers' purchasing decisions (Chen et al., 2022), so we would be remiss if we were to neglect them. Finally, we will scale this combination of income and sentiment by some factor representing the rate at which a listing is booked.

From this, we may take the success of a listing to be the product $S=r\cdot p\cdot s_\mu$. 

The daily rate $r$ is the advertised price of the listing (which we must assume to be the price ultimately paid by any tenants), given in the local currency (e.g. GBP, USD, etc.). This rate can vary wildly, and the significance of an increase in rate diminishes as the baseline rate increases (for example, increasing the daily rate of a property from £50 to £100 does not carry the same weight as increasing from £500 to £550, even though they are both the same absolute increase). Hence, we should apply some transformation that emphasises changes in lower daily rates and reduces the weight of larger absolute changes in higher daily rates that reflect a small percentage change. An appropriate transformation here would be a log transformation, which also serves as handling of skewness, which already appears in real estate pricing. This is the approach taken in Lawani et al. (2019).

The probability $p$ that a property is booked out on any given day is difficult to infer from our data, so we will discuss this separately. 

The average review sentiment $s_\mu$ can be interpreted in several ways. Most easily, we could determine the sentiment of all reviews and find the mean sentiment. This might be too simplistic for our problem, in which we have properties that may have undergone significant change over time or listings that have a noteworthy trend in their reviews over time (e.g. increasingly negative). A more considerate approach, then, is to take a weighted average, with greater weight being given to more recent reviews (we might fix the weighting by time, or we may rank reviews by recency and weigh them in proportion). 

With this considered, we further specify:
$$
S=\frac{p\ln(r)}{|R|}\cdot\sum_{s\in R}s\cdot w(s)
$$
where $R$ is the set of reviews and $w(s)$ is a function specifying the appropriate weight for a given review sentiment $s$ (determined with respect to the recency of $s$).

### Sentiment Analysis on Raw Textual Data
We will use the Python library [TextBlob](https://textblob.readthedocs.io/en/dev/) for all of our NLP concerns, sentiment analysis included, as with Kalehbasti et al. (2021).

There are several sources of raw textual data associated with Airbnb listings. The primary sources of interest for us are the listing's title, the description of the property, the description of neighbourhood, the about section for the host, and the reviews left by previous tenants.

For each of these sources, we will conduct sentiment analysis to obtain a **polarity** score and a **subjectivity** score.
	**Polarity** denotes how positive or negative the sentiment of the text is deemed to be, with -1 representing absolutely negative, 0 representing neutral, and 1 representing absolutely positive.
	**Subjectivity** denotes how likely the text is to be factual or opinion based; a score of 0 represents an absolutely objective view, and 1 represents absolutely subjective.

In addition to this, we might find it interesting to find the most 'successful' keywords picked out from titles, descriptions, and so on. We would do this by using keyword extraction to identify the most prominent keyword from whatever textual data source, and note the success score of the corresponding listing. Taking the average score for each keyword would infer which keywords are the most 'successful' in online real estate marketing (at least on Airbnb).

### Computing the occupancy rates of listings
Now we turn our attention to the details of the probability $p$ that a property is rented out on any given day. This is a rephrasing of the more commonly discussed **occupancy rate**, which is computed with a plethora of asterixis.

We take our method for estimating the occupancy rate of listings, and hence the probability $p$, from Marqusee (2015). We make modifications in line with [Inside Airbnb's](http://insideairbnb.com/data-assumptions/) "San Francisco Model", which involves taking a review rate of 50% (instead of the unverifiable 72% claim or the overly conservative estimate of 30.5%).

Further, we dislike taking an average length of stay to be used with all listings, as it ignores pricing strategies that take longer or shorter stays into account and adjust price accordingly. As such, we will take the average 'minimum night' value for each listing (over the last twelve months) to be our conservative estimate for the length of stay for each booking of the listing.
	This should allow properties intended for 'longer' duration stays that market with a higher 'minimum night' value to have a more representative estimate.
