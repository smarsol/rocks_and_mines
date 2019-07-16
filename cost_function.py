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

def calcul_cost(tn, fn, tp, fp, total, cost_fn, cost_fp):
  return (tn * 200 + fn * cost_fn + tp * 1000 + fp * cost_fp)/total

def cost(prediction, real, cost_fn, cost_fp):
  cost_total = []
  gammas = np.arange(-0.5, 1.5, 0.1)
  total = len(real)
  for gamma in gammas:
    tp, fp, tn, fn = confusion_matrix(prediction, real, gamma)
    cost_total.append(calcul_cost(tn, fn, tp, fp, total, cost_fn, cost_fp))
  return cost_total



data = pd.read_csv('sonar.all-data', header = None)
train, test = train_test_split(data, test_size = 0.2)

train_y = train.iloc[:,-1].apply(lambda x: 1 if x == 'M' else 0)
train_x = train.iloc[:,:-1]
model = LinearRegression().fit(train_x, train_y)
test_x = test.iloc[:,:-1]
test_yhat = model.predict(test_x)
test_y = test.iloc[:,-1].apply(lambda x: 1 if x == 'M' else 0)
train_yhat = model.predict(train_x)

cost_fns = [1000, 2000]
cost_fps = [1500, 2500]
gammas = np.arange(-0.5, 1.5, 0.1)
for cost_fn in cost_fns :
  for cost_fp in cost_fps:
    plt.plot(gammas, cost(test_yhat, test_y, cost_fn, cost_fp), label = "cost_fn = %d and cost_fp = %d"%(cost_fn, cost_fp))
plt.legend()
plt.show()
