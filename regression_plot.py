import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split


def confusion_matrix(predicted, real, gamma):
  test_decisions = np.where(predicted < gamma, 0, 1)  
  tp = np.logical_and(test_decisions == 1, real == 1).sum()
  fp = np.logical_and(test_decisions == 1, real == 0).sum()
  tn = np.logical_and(test_decisions == 0, real == 0).sum()
  fn = np.logical_and(test_decisions == 0, real == 1).sum()
  return tp, fp, tn, fn

def false_positives(model, test):
  gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  llista = []
  for x in gamma:
    test_x = test.iloc[:,:-1]
    test_yhat = model.predict(test_x)
    test_y = test.iloc[:,-1].apply(lambda x: 1 if x == 'M' else 0)
    test_decisions = np.where(test_yhat < x, 0, 1)
    llista.append(np.logical_and(test_decisions == 1, test_y == 0).sum())
  plt.scatter(gamma, llista)
  plt.show()
  return llista, gamma

data = pd.read_csv('sonar.all-data', header = None)
train, test = train_test_split(data, test_size = 0.2)

train_y = train.iloc[:,-1].apply(lambda x: 1 if x == 'M' else 0)
train_x = train.iloc[:,:-1]

print (train_x.head())
model = LinearRegression().fit(train_x, train_y)
test_x = test.iloc[:,:-1]
test_yhat = model.predict(test_x)
test_y = test.iloc[:,-1].apply(lambda x: 1 if x == 'M' else 0)
gamma = 0.25
test_decisions = np.where(test_yhat < gamma, 0, 1)

print (test_yhat)
print (test_y)
print (np.abs((test_yhat - test_y) ** 2))
print (test_decisions)
print (model.score(test_x, test_y))

train.head()
test.head()

# print(false_positives(model, test))
print(confusion_matrix(test_yhat, test_y, gamma))

train_yhat = model.predict(train_x)

tpr_train = []
fpr_train = []
fpr_test = []
tpr_test = []

for gamma_test in np.arange(0, 1, 0.01):
  tp_train, fp_train, tn_train, fn_train = confusion_matrix(train_yhat, train_y, gamma_test)
  tpr_train.append(tp_train/(tp_train + fn_train))
  fpr_train.append(fp_train/(tn_train + fp_train))
  tp_test, fp_test, tn_test, fn_test = confusion_matrix(test_yhat, test_y, gamma_test)
  tpr_test.append(tp_test/(tp_test + fn_test))
  fpr_test.append(fp_test/(tn_test + fp_test))

print(tpr_test)
print(fpr_test) 

fig = plt.figure()
plt.plot(fpr_train, tpr_train, "o--", label = 'train')
plt.plot(fpr_test, tpr_test, "o--", label = 'test')
plt.legend()
plt.xlabel("FPR", fontsize = 18)
plt.ylabel("TPR", fontsize = 18)
fig.tight_layout()
plt.show()
fig.savefig("roc_test_train.png")

