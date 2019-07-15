import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('sonar.all-data', header = None)
train, test = train_test_split(data, test_size = 0.2)

train_y = train.iloc[:,-1].apply(lambda x: 1 if x == 'M' else 0)
train_x = train.iloc[:,:-1]

print (train_x.head())

rm_linear = LinearRegression().fit(train_x, train_y)

test_x = test.iloc[:,:-1]
test_yhat = rm_linear.predict(test_x)

train.head()
test.head()

# f_4 = data.iloc[:,10]
# f_24 = data.iloc[:,33]
# labels = data.iloc[:,-1]

# plt.scatter(f_4, f_24, c = labels)
# plt.show()

