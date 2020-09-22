# AAPECS
## Machine learning (ML) pipline for AAPECS
This ML pipline contains the following modules: data cleaning, missing data imputation, feature extraction and machine learning analysis. The used imputation method is multiple imputation by chained equations (MICE). Extracted features includes intra-day feature and inter-day features. The machine learning analysis is within the framework of Leave-one-participant-out cross-validation.
* data_path store the raw data
* result_path store the output results
* iter is the number of iteration in mice imputation
* thres is the threshold for correlation feature selection in linear regression model
