# AAPECS
## Machine learning (ML) pipline for Predicting Adaptive and Maladaptive Traits
We built two types of modeling: day-level modeling and person-level modeling. For the day-level modeling, the input is daily features and one partipants have multiple values for one daily feature. For the person-lvel modeling, we aggreate multiple daily values into one person feature, which is the input of the ml learning pipline. Two modeling processes share the same ML pipeline. The ML pipeline includes three basic ML models: Lasso regression (Lasso), Random Forest regression (RF), and XGBoost regression (XGB). The fundamental training process is leave-one-particiapnt-out cross-validation, and the evaluation metrics includes MSE and MAE.
* ## Step 1: Configure the parameter and directory.
* ### Go to param.py to set the reading path of dataset and thresholds used for cleaning dataset and selecting features.
* ### Go to ml_param.py to set the hyper-parameter space used for tuning parameters.
* ## Step 2: clean the dataset.
* ### For the day-level modeling, run pre_process_day.ipynb.
* ### For the person-level modeling, run pre_process_person.ipynb.
* ## Step 3: Run the ML pipline.
* ### For the day-level modeling, run the day_model_lasso.ipynb, day_model_rf.ipynb, day_model_xgb.ipynb
* ### For the person-level modeling, run the person_model_lasso.ipynb, person_model_rf.ipynb, person_model_xgb.ipynb
* ## Step 4: Visualize the evaluatiion of ML performance. 
* ### Run the visualization.ipynb to plot the radar plots of MSE and MAE for day-level and person-level modeling respectively.
* ## Step 5: Ranking sensor features and analyzing the correlation between sensor feature and personality Trait.
* ### Run the heatmap.ipynb to get the matrix of ranking scores and correlation
* ### Run the heatmap.py to get the Figure 8 in the paper. 
* #### (* ml_models.py contains code of training ml models, imputation.py contains code of filling missing values, pre_process.py contains code of cleaning dataset)
