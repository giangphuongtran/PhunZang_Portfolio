import numpy as np
import pandas as pd


data = pd.read_csv('AUSweather.csv')

data.columns = [i.lower() for i in data.columns]
data = data.dropna(subset= ['raintomorrow'], axis= 0)
data = data.drop('date', axis= 1)

null_value = pd.DataFrame((data.isnull().sum().sort_values(ascending= False)*100/len(data)).reset_index())
null_value_to_drop = list(null_value['index'].head(4))
data = data.drop(null_value_to_drop, axis= 1)
data = data[data.isnull().sum(axis= 1) < 5]
data['month'] = pd.to_datetime(data['date']).dt.month
data = data.drop('date', axis= 1)


data['raintomorrow'] = [1 if i == 'Yes' else 0 for i in data['raintomorrow']]
input_var = data.drop('raintomorrow', axis= 1)
target_var = data['raintomorrow']


from sklearn.compose import ColumnTransformer, make_column_selector # Transformer different method on different types of variables
from sklearn.impute import SimpleImputer # Impute on null - values
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


num_col = make_column_selector(dtype_include= 'number')
cat_col = make_column_selector(dtype_exclude= 'number')
logreg = LogisticRegression()


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy= 'median')),
    ('scaler', StandardScaler()),
])


cat_pipeline = Pipeline([
    ('simpleimputer', SimpleImputer(strategy= 'constant', fill_value= 'Unknown')),
    ('ohe', OneHotEncoder(categories= 'auto', handle_unknown= 'ignore', sparse= False)),
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_col),
        ('cat', cat_pipeline, cat_col)
])


full_pipe = Pipeline(
    steps=[
        ('pp', preprocessor),
        ('classifier', logreg),
])



full_pipe.fit(input_var, target_var)


import pickle
model_filename = 'weather-model.pkl'
pickle.dump(full_pipe, open(model_filename,'wb'))


model = pickle.load(open('weather-model.pkl','rb'))


print(model.predict(pd.DataFrame({'location': {0: 'Albury'},
 'mintemp': {0: 13.4},
 'maxtemp': {0: 22.9},
 'rainfall': {0: 0.6},
 'windgustdir': {0: 'W'},
 'windgustspeed': {0: 44.0},
 'winddir9am': {0: 'W'},
 'winddir3pm': {0: 'WNW'},
 'windspeed9am': {0: 20.0},
 'windspeed3pm': {0: 24.0},
 'humidity9am': {0: 71.0},
 'humidity3pm': {0: 22.0},
 'pressure9am': {0: 1007.7},
 'pressure3pm': {0: 1007.1},
 'temp9am': {0: 16.9},
 'temp3pm': {0: 21.8},
 'raintoday': {0: 'No'}})))