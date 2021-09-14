import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# sns.set()



DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("data", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()



def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def get_data_info():
    print(housing.info())
    data_describe = housing.describe()
    return data_describe


fetch_housing_data()
housing = load_housing_data()
data_describe = get_data_info()
housing = pd.get_dummies(housing)

plt.figure(1)
sns.histplot(data=housing)

plt.figure(2)
sns.scatterplot(data=housing, x='longitude', y='latitude', hue='median_house_value', s=1)


corr_matrix = housing.corr()

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

target = housing.pop('median_house_value')
housing['median_house_value'] = target

housing = housing.dropna()
corr_matrix_2 = housing.corr()

X = housing.loc[:, 'longitude':'population_per_household']
y =  housing['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y)

def evaluate_model(estimator):
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_train)
    print('\nAccuracy on train set:', model.score(X_train, y_train))
    mse = np.sqrt(mean_squared_error(y_train, y_predicted))
    print('Mean squared error:', mse)

    print('Accuracy on test set:', model.score(X_test, y_test))

estimators = [LinearRegression(), RandomForestRegressor()]
for i in estimators:
    evaluate_model(i)


plt.show()
