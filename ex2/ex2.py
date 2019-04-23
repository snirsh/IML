import pandas as pd
import numpy as np
import numpy.linalg as lin

df = pd.read_csv('kc_house_data.csv')
df = df.drop(['date', 'id', 'lat', 'long'], axis=1)  # remove columns that i dont need
df.dropna()  # remove all nulls

# cleaning noise

df = df[df['price'] > 0]
df = df[df['zipcode'] > 0]
df = df[df['sqft_living'] > 0]
df = df[df['price'] > 0]
df = df[df['sqft_lot'] > 0]
df = df[df['yr_built'] >= 1900]
df = df[df['yr_built'] <= 2019]
df = df[df['bathrooms'] > 0]
df = df[df['bedrooms'] >= 1]
df = df[df['bedrooms'] <= 10]

# averages to calculate
price_average = df['price'].mean()
sqft_live_average = df['sqft_living'].mean()
sqft_lot_average = df['sqft_lot'].mean()
sqft_above_average = df['sqft_above'].mean()

df = df[df['price'] <= price_average * 7]
df = df[df['price'] >= price_average / 7]
df = df[df['sqft_living'] <= sqft_live_average * 2]
df = df[df['sqft_living'] >= sqft_live_average / 2]
df = df[df['sqft_lot'] <= sqft_lot_average * 2]
df = df[df['sqft_lot'] >= sqft_lot_average / 3]
df = df[df['sqft_above'] <= sqft_above_average * 2]
df = df[df['sqft_above'] >= sqft_above_average / 3]

# make zipcode a dummy variable
df.loc[df['zipcode'] > 98100, 'zipcode'] = 1
df.loc[df['zipcode'] > 1, 'zipcode'] = 0

# Make X and Y
X = df.drop(['price'], axis=1)
Y = df[['price']]

theta = np.zeros((df.shape[1], 1))
s_theta = np.zeros((df.shape[1], 1))
mb_theta = np.zeros((df.shape[1], 1))


def h(theta, X):
    tempX = np.ones((X.shape[0], X.shape[1] + 1))
    tempX[:, 1:] = X
    return np.matmul(tempX, theta)


def loss(theta, X, Y):
    return np.average(np.square(Y - h(theta, X))) / 2

