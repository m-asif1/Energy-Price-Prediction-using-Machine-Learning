# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# import csv file
energy_data = pd.read_csv('energy_dataset.csv')

# check head of file data
energy_data.head()

# check tail of file data
energy_data.tail()

# check the shape of data
energy_data.shape

# check the columns of data
energy_data.columns

# check the info of data
energy_data.info()

energy_data.describe()

# check the null values of data
energy_data.isnull().sum()

# Drop the columns
energy_data = energy_data.drop(['generation hydro pumped storage aggregated',
'forecast wind offshore eday ahead'], axis = 1)

# Drop nan values
energy_data = energy_data.dropna()
energy_data.isnull().sum()

# Check the unique values
energy_data.nunique()

energy_data = energy_data.drop(['time'], axis = 1)

# Round the values
round((energy_data.isnull().sum()/len(energy_data)*100),2)

# check the correlations
energy_data.corr()

correlations = energy_data.corr(method='pearson')

# print the corelation
print(correlations['price actual'].sort_values(ascending=False).to_string())


null_val_cols = ['generation marine',
'generation geothermal',
'generation fossil peat',
'generation wind offshore',
'generation fossil oil shale',
'generation fossil coal-derived gas']

heat_map_features = energy_data.drop(columns=null_val_cols,axis=1)

# plot of heat map
plt.figure(figsize=(15,12.5))
sns.heatmap(round(heat_map_features.corr(),1),annot=True,
cmap='Blues',linewidth=0.9)
plt.show();


# plotting histogram
plt.figure(figsize=(15,10))
sns.histplot(energy_data,x='price actual');
plt.show();

# plotting scatterplot
sns.scatterplot(x='total load actual',y='price actual',
data = energy_data)

x = energy_data.drop(['price actual'], axis = 1)
y = energy_data['price actual']

# Linear Regression
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.linear_model import Ridge, LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)
y_pred = model.predict(xtest)

print("Training Accuracy :", model.score(xtrain, ytrain))
print("Testing Accuracy :", model.score(xtest, ytest))
# Training Accuracy : 0.6180465791911046
# Testing Accuracy : 0.619945448043943

# Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(xtrain, ytrain)
y_pred = regressor.predict(xtest)
print("Training Accuracy :", regressor.score(xtrain, ytrain))
print("Testing Accuracy :", regressor.score(xtest, ytest))
# Training Accuracy : 0.9817078228271837
# Testing Accuracy : 0.8825272519289389