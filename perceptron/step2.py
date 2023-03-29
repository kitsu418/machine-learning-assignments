# encoding=utf8
from sklearn.linear_model.perceptron import Perceptron
import os

if os.path.exists('./step2/result.csv'):
    os.remove('./step2/result.csv')

#********* Begin *********#
import pandas as pd
train_data = pd.read_csv('./step2/train_data.csv')
train_label = pd.read_csv('./step2/train_label.csv')
train_label = train_label['target']
test_data = pd.read_csv('./step2/test_data.csv')

clf = Perceptron(eta0=1.0, max_iter=8)
clf.fit(train_data, train_label)
result = clf.predict(test_data)

df = pd.DataFrame({"result": result})
df.to_csv("./step2/result.csv")
#********* End *********#
