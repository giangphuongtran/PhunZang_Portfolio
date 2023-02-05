# Predicting Australia weather using Classification Algorithms

With a dataset given 10 years of daily weather observations across Australia, we are going to predict whether it will rain tomorrow or not.
> + In Albury, with some kind of weather indicator record, should we hangout the next day?

### [Data Source from Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

![Map of Australia](https://github.com/giangphuongtran/PhunZang_Portfolio/blob/main/Australia_weather/australia_map.jpeg)

## [Detail of the notebook:](https://github.com/giangphuongtran/PhunZang_Portfolio/blob/main/Australia_weather/aus_weather.ipynb)

**1. Import needed library**
  - Library for Data Pre-processing & Analysis
  - Library for Modelling

**2. Data Pre-Processing**
  + Load the Australia weather dataset
  + Some exploratory analysis
    + Drop high-percentage null values before doing any analysis and processing
    + Statistical look
      + Outlier detector using IQR
      + Other variables distribution
  + Feature Engineering & Selection

**3. Modelling**
  + Initial Model
  + Hyperparameter Tuning With GridSearchCV and Cross Validation
  + Increase Accuracy (Coming soon)

**4. Model Deployment using Flask API**
