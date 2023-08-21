# Analysis of Machine Learning Models for Predicting Loan Risk Status

## Overview of the Analysis

The purpose of this analysis was to develop and evaluate models for predicting loan categories based on several input features. The loan categories were healthy and high-risk, and the input features were loan size, interest rate, borrower income, debt to income ratio, number of accounts, number of derogatory remarks, and total ammount of debt. The healthy loans were assigned a category of 0, and the high-risk were assigned 1. The dataset that was provided had a high percentage of healthy compared to high-risk, as that is most likely the distribution encountered in the loan market. The dataset included a total of 77536 observations, of which 75036 were healthy loans, and the reamining 2500 were high-risk. 

The data were imported and stored in a Pandas dataframe, then the y-lables were set as the loan status for each observation, and the input features were assigned to the dataframe with the loan status column dropped. The data were then separtated into training and testing subsets, using a fraction of 0.25 as the training dataset. Since this was a classification task, a logistic regression model was used. The training data were used to train the model, then the testing data were used to generate predictions that were compared to the true y-labels. The accuracy score, confusion matrix, and classification report were used to evalute the model. 

Since the dataset conatined a high percentage of healthy loans, an oversampling technique was used to retrain the logistic regression model. This entailed using a random oversampling algorithm that equalized the number of healthy and high-risk observations by sampling the high risk obseravtions with replacement. This resulted in a dataset that contained 56271 oberstations of both healthy and high-risk loans. The model was then retrained with the oversampled dataset using the same train/test ratio, and evaluted in the same manner as above.

## Results
  
Machine learning model 1 - Logistic Regression

* Healthy loans
    * Total number of loans: 18765; total healthy predictions: 18719; high-risk loans classfied as healthy: 56
    * Precision: 0.997; Recall: 0.99; F-1: 0.994
    
* High-Risk loans
    * Total number of loans: 619; total high-risk predictions: 665; healthy loans classified as high-risk: 102
    * Precision: 0.85; Recall: 0.91; F-1: 0.879
    
* Accuracy of machine learning model 1: 0.99

Machine learning model 2 - Logistic regression with oversampling

* Healthy loans
    * Total number of loans: 18765; total healthy predictions: 18649; high-risk loans classfied as healthy: 4
    * Precision: 1; Recall: 0.99; F-1: 1
    
* High-Risk loans
    * Total number of loans: 619; total high-risk predictions: 615; healthy loans classified as high-risk: 116
    * Precision: 0.84; Recall: 0.99; F-1: 0.91
    
* Accuracy of machine learning model 1: 0.99

## Summary

The first machine learning model predicted 18663 of 18765 healthy loans, resulting in a recall score of 0.99, and of the 18719 healthy loan predictions, 56 of them were actually applied to high-risk loans, resulting in a precision score of 0.997. The F-1 score for healthy loans was 0.994. The model predicted 563 of the 619 high-risk loans, resulting in a recall score of 0.91, and of the 665 high-risk loan predictions, 102 of them were applied to healthy loans, resulting in a precision score of 0.85. The F-1 score for high-risk loans was 0.879. In summary, there was a higher chance of correctly classfying a healthy loan than there was of correctly classfying a high-risk loan, and there was a higher chance of falsely classifying a healthy loan as high-risk than there was of classifying a high risk loan as healthy.

Using an oversampling technique, the resulting model predicted 18649 of 18765 healthy loans, resulting in a recall score of 0.99, and of the 18653 healthy loan predictions, only 4 of them were actually applied to high-risk loans, resulting in a precision score of nearly 1. The F-1 score for healthy loans was very close to 1. The model predicted 615 of the 619 high risk loans, resulting in a recall score of 0.99, and of the 731 high-risk loan predictions, 116 of them were applied to healthy loans, resulting in a precision score of 0.84. The F-1 score for high-risk loans was 0.91. In summary, using an oversampling technique, the model improved the recall and F-1 scores for both categroies of loans while sacrificing some amount of precision when identifying high-risk loans, resulting in a slightly higher rate of falsely classfying healthy loans as high-risk.

Overall, the model choice depends on whether the lending institution wishes to issue more loans in order to capture a higher percentage of low-risk customers while sacrificing recall of high-risk customers, or if it wishes to be more discriminatory of high-risk customers while sacrificing some recall of low-risk customers. It seems that the second model greatly improves both the recall and F-1 scores for the high-risk customers, improves the precision and F-1 scores for low-risk customers, and only sacrifices a small ammount of the precision for high-risk customers. Therefore, I would recommend the second model based on its overall performance. One other factor I would explore is whether or not to include the debt to income ratio, as this is calculated from two other input features namely the borrower's income and total debt. This introduces multicollinearity into our model.
