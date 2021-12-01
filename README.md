# AAPECS
## Machine learning (ML) pipline for Predicting Adaptive and Maladaptive Traits
We built two types of modeling: day-level modeling and person-level modeling. For the day-level modeling, the input is daily features and one partipants have multiple values for one daily feature. For the person-lvel modeling, we aggreate multiple daily values into one person feature, which is the input of the ml learning pipline. Two modeling processes share the same ML pipeline. The ML pipeline includes three basic ML models: Lasso regression (Lasso), Random Forest regression (RF), and XGBoost regression (XGB). The fundamental training process is leave-one-particiapnt-out cross-validation, and the evaluation metrics includes MSE and MAE.
* Step 1: Configure the parameter and directory.
* # 
