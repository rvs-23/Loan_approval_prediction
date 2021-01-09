# Loan approval prediction

### Project Overview
<pre>
Loans allow for growth in the overall money supply in an economy and open up competition by
lending to new businesses.1 Thus, with the growing pace of emerging businesses, speeding up of the
approval process for potential customers becomes very important.
In this project, I attempted to build a Machine Learning model based on 7 features that is capable of
predicting whether an individual should be eligible for loan approval.
</pre>


### Problem Statement
<pre>
The goal is to create a classifier that can predict whether an individual should be approved for loan or
not; the tasks involved are the following:
i) Download and pre-process the data.
ii) Gain insights about the data, engineer features (If necessary) and make it ready
for the machine learning algorithms.
iii) Train the data and test for evaluation metrics.
iv) Fine tune the hyperparameters.
v) Make predictions on the testing set.
</pre>


### Metrics
<pre>
The most common metric for a binary classifier is Accuracy but it is not as robust as some other
metrics like Balanced Accuracy, AUC-ROC. Therefore, we can use the following metrics to judge the
performance of this classifier:
i) Balanced Accuracy (BAC) = 0.5*(Sensitivity + Specificity)
ii) Recall (Sensitivity) = Tells us what % of True positives were recalled.
iii) AUC = Area under curve of a ROC curve.
</pre>


### Exploratory Data Analysis
<pre>
Since the number of features were small, we can begin the process of exploring the data feature by
keeping these three questions in mind:
i) What does the distribution of each feature look like?
ii) If the variables are categorical, how much does each category impact the Loan status?
iii) If there are missing values, what should be the best strategy to impute them?
</pre>

### Data Wrangling and Feature Engineering
<pre>
1. The data set has 8 categorical features and 3 continuous features.
2. Binary features like Gender, Married, Education, Self-employed, Credit-History, Loan status
were all One Hot encoded. Example: Gender was encoded 1 if the applicant was male, 0
otherwise.
3. A person with 1 dependent had the highest chance of getting their loan approved. Therefore,
while encoding maximum weight was given to applicants with 1 dependent.
4. Applicant Income had outliers. So, discretization of Applicant Income, based on their sample
quantiles, into 5 categories was a valid solution.
5. Loan Amount was almost Normally distributed but it had few outliers which we had to deal
with. Therefore, after filling the null spaces with the median of the distribution, the feature
Loan Amount was also, based on the sample quantiles, discretized into 5 categories.
6. Loan Amount term had only 10 unique values in the form of the number of years. Therefore, it
was label encoded.
7. EDA of Property Area showed that individuals living in semi-urban areas were more likely to get
their loans approved. Therefore, most weight was given to the applicants coming from semiurban areas.
8. Finally, the last feature i.e., Co-applicant Income. This feature provided some pretty interesting
insights as mentioned in the EDA part. Because the increase in co-applicant income did not play
a big role in increasing the chances of approval of an applicant combined with the fact that 44%
of the values were exactly 0, we can engineer a new feature and call it “co-applicant income
exists” which is 1 if the co-applicant income is non-zero, 0 otherwise.
</pre>



### Algorithms
<pre>
The algorithms used in this project are the following:
i) Random Forests
ii) Logistic Regression
iii) Support Vector Classifier
iv) Gaussian Naïve Bayes2
2 The Optimality of Naive Bayes (aaai.org)
RISHAV SHARMA 14
The number of observations in the training set were pretty small so it made more sense to choose
algorithms with a high bias and low variance (ii, iii, iv)
Random Forests serve as a comparison model to see how they perform against the other three.
• Initially, all 11 features were used to evaluate the performance of the algorithms.
• Once, we got the top 2 performing models from the first iteration, their feature importance
values were used to create a second feature set
• Finally, a third feature set was created based on the correlation analysis of all 11 features. The
features that were least corelated with Loan status were dropped and the ones that were
highly corelated among themselves were also taken care of.
</pre>


### Result
<pre>
Judgement: Both Recall and Balanced accuracy are our evaluation metrics. Looking at the final table, I
would choose LRC_f2 or Logistic Regression with feature set 2 to be the model for making predictions
on the testing data.
Reason: The balanced accuracy score of the topmost model is 0.742 which is higher when compared
to 0.723; the balanced accuracy score of the model of my choice. But the increase is a mere 2% and for
a small testing set of size 347, this increase is pretty insignificant particularly when we factor in the dip
in Recall score values which is about 11%.
Therefore, I made the predictions on the testing set using a logistic regression model with feature set
2.
</pre>
