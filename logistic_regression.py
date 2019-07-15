import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('sonar.all-data', header = None)
train, test = train_test_split(data, test_size = 0.2)

train_y = train.iloc[:,-1].apply(lambda x: 1 if x == 'M' else 0)
train_x = train.iloc[:,:-1]

print (train_x.head())
model = LogisticRegression().fit(train_x, train_y)

test_x = test.iloc[:,:-1]
test_yhat = model.predict(test_x)
test_y = test.iloc[:,-1].apply(lambda x: 1 if x == 'M' else 0)
gamma = 0.25
test_decisions = np.where(test_yhat < gamma, 0, 1)

print (test_yhat)
print (test_y)
print(np.abs((test_yhat - test_y) ** 2))
print (test_decisions)
print (model.score(test_x, test_y))

train.head()
test.head()

TP = np.logical_and(test_decisions == 1, test_y == 1).sum()
FP = np.logical_and(test_decisions == 1, test_y == 0).sum()
TN = np.logical_and(test_decisions == 0, test_y == 0).sum()
FN = np.logical_and(test_decisions == 0, test_y == 1).sum()

print('True positive', TP)
print('False positive', FP)
print('True negative', TN)
print('False negative', FN)
print('Gamma', gamma)


# f_4 = data.iloc[:,10]
# f_24 = data.iloc[:,33]
# labels = data.iloc[:,-1]

# plt.scatter(f_4, f_24, c = labels)
# plt.show()

