BMDS2003 - Malaysia Housing Median Price Estimator
==================================================
MODEL CONTRACT
  State, Area_Key, Tenure, Primary_Type, Median_PSF, Transactions
  Median_PSF must be independently known before prediction.

SELECTED MODEL
  Random Forest selected by GroupKFold on the training set. Models within
  1% of the best mean are treated as a practical tie; lower fold variation wins.

CONTENTS
  Latest21.ipynb                               full CRISP-DM notebook
  malaysia_house_price_data_2025.csv           raw dataset
  malaysia_house_price_cleaned_with_area.csv   cleaned dataset
  malaysia_house_price_model_ready.csv         unencoded model input table
  model_results.csv                            full model metrics
  model_comparison_table.csv                   clean model comparison table
  tuning_results.csv                           parameter tuning summary
  fold_scores.csv                              RMSE, MAE and R2 for each fold
  median_psf_ablation_results.csv              Median PSF dependency evidence
  test_predictions.csv                         hold-out predictions
  area_preprocessing.py                        shared Area cleaning module
  streamlit_app.py                             deployment prototype
  .streamlit/config.toml                       light theme
  models/                                      4 official trained pipelines (.pkl)
  figures/                                     Figures 1-31
  assets/                                      optional hero banner
  requirements.txt

TO RUN THE PROTOTYPE
  1. Keep every file above in one folder.
  2. pip install -r requirements.txt
  3. streamlit run streamlit_app.py
