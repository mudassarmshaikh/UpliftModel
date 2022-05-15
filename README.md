# Campaign Conversion Target with Uplift Modeling

### Uplift modeling, also known as incrementality modeling or persuasion modeling, models the causal effect of a treatment on the outcome for different individuals. Uplift modeling has been widely applied in marketing, personalized medicine, and political elections, scenarios in which you don’t simply want to predict an outcome, but predict how that outcome might change with respect to a treatment.

*Uplift modeling is an important yet novel area of research in machine learning which aims to explain and to estimate the causal impact of a treatment at the individual level.*
<p align="center">
  <img src="/uplift-treatmentvsresponse.jpg" width="400" height="400">
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
