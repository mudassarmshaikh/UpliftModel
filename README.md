# Campaign Conversion Uplift Probability

### Uplift modeling, also known as incrementality modeling or persuasion modeling, models the causal effect of a treatment on the outcome for different individuals. Uplift modeling has been widely applied in marketing, personalized medicine, and political elections, scenarios in which you don’t simply want to predict an outcome, but predict how that outcome might change with respect to a treatment.

*Uplift modeling is an important yet novel area of research in machine learning which aims to explain and to estimate the causal impact of a treatment at the individual level.*
<p align="center">
  <img src="/uplift-treatmentvsresponse.jpg" width="500" height="400">
 </p>

In the digital advertising industry, the treatment is exposure to different ads and uplift modeling is used to direct marketing efforts towards users for whom it is the most efficient. One of the main applications of the Machine Learning models is to improve Targeted Marketing. Targeted Marketing is used to select the customers that most likely buy a product.

There are different approaches for Targeted Marketing such as the Classical modeling (also known as Response model). This approach is focused on training a model with customers that were sent a promotion/offer. The model separates the customers that will buy from the ones that will not buy if targeted. This approach generates higher profit than random selection targeting.

Classical modeling has a flaw. It does not separate customers that will buy even if not targeted from the ones that will only buy if targeted. In other words, the model wastes money by targeting customers that do not need it. The current solution to that issue is the Uplift modeling.

Uplift modeling can separate customers that will buy if not targeted from the ones that will buy only if targeted, as well as avoiding customers that will not buy even if targeted. Specifically, the model identifies the customers that are worth spending money on Targeted Marketing.

#### When should we use uplift modeling?
Uplift modeling is used when the customer’s target action is likely to happen without any communication. For instance, we want to promote a popular product but we don’t want to spend our marketing budget on customers who will buy the product anyway with or without communication. If the product is not popular and it has to be promoted to be bought, then a task turns to the response modeling task

The Uplift model evaluates the net effect of communication by trying to select only those customers who are going to perform the target action only when there is some advertising exposure presenting to them. The model predicts a difference between the customer’s behavior when there is a treatment (communication) and when there is no treatment (no communication).

Uplift modeling, also known as incremental modeling, true lift modeling, or net modeling is a predictive modeling technique that directly models the incremental impact of a treatment (such as a direct marketing action) on an individual's behavior.
The goal of uplift modeling, also known as net lift or incremental response modeling,
- is to identify the “persuadables”,
- not waste efforts on “sure things” and “lost causes”,
- and avoid bothering “sleeping dogs”, or those who would react negatively to the treatment, if they exist.

Let's generate a model that can identify users that are more likely to convert (or buy the product) and avoid the ones that are not.

### Uplift modeling using Advertising Data
We are working with a dataset that is constructed by assembling data resulting from several incrementality tests, a particular randomized trial procedure where a random part of the population is prevented from being targeted by advertising. It consists of 13M rows, each one representing a user with 11 features, a treatment indicator and 2 labels (visits and conversions).

Dataset is available at https://ailab.criteo.com/criteo-uplift-prediction-dataset/

The dataset is a collection of 13 million samples from a randomized control trial, scaling up previously available datasets by a healthy 590x factor.
The data was provided by AI lab of Criteo (French advertising company that provides online display advertisements). The data contains 13 million instances from a randomized control trial collected in two weeks, where 84.6% of the users where sent the treatment.

Each instance has 12 features that were anonymized plus a treatment variable and two target variables (visits and conversion). There is another extra variable called "exposure" which indicates whether the user was effectively exposed to the treatment. The dataset consists of 13M rows, each one representing a user with 12 features, a treatment indicator and 2 binary labels (visits and conversions). Positive labels mean the user visited/converted on the advertiser website during the test period (2 weeks). The global treatment ratio is 84.6%. It is usual that advertisers keep only a small control population as it costs them in potential revenue. Following is a detailed description of the features:

- f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11: feature values (dense, float)
- treatment: treatment group (1 = treated, 0 = control)
- conversion: whether a conversion occurred for this user (binary, label)
- visit: whether a visit occurred for this user (binary, label)
- exposure: treatment effect, whether the user has been effectively exposed (binary)

**There are two target variables (visits and conversion), this project will only focus on the conversion variable, which can be understood as the indicator whether the user bought the product.
The goal is to generate a model that can identify users that are more likely to convert (or buy the product) and avoid the ones that are not.**


**Download and host the dataset in your environment** http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz

## Run the [Jupyter notebook to perform EDA](https://github.com/mudassarmshaikh/UpliftModel/commit/52d996ddb394510432784d99d2d60b9b5dff9ae6).

## Hyperparameter Tuning

# scikit-uplift

**scikit-uplift (sklift) is an uplift modeling python package that provides fast sklearn-style models implementation, evaluation metrics and visualization tools.
The main idea is to provide easy-to-use and fast python package for uplift modeling. It delivers the model interface with the familiar scikit-learn API. One can use any popular estimator (for instance, from the Catboost library).
Uplift modeling estimates a causal effect of treatment and uses it to effectively target customers that are most likely to respond to a marketing campaign.**

## Implementing Two Model Approach
The main idea is to estimate the conditional probabilities of the treatment and control groups separately.
- Train the first model using the treatment set.
- Train the second model using the control set.
- Inference: subtract the control model scores from the treatment model scores.

# Train the Model

# Host API to test with Postman
