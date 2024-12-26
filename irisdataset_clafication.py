from sklearn import datasets 
import numpy as np 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
iris = datasets.load_iris()
# print (list(iris.keys()))
x = iris['data'][: ,3:]
y = (iris['target'] == 2).astype(np.int_)
# print (x,y) 
clf = LogisticRegression()
clf.fit(x,y)
example = clf.predict(([[2.6]]))
print(example)
x_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(x_new)
plt.plot(x_new,y_prob[:,1], "g-",label="verginica")
plt.show()
# print (iris['data'])
# print(iris['target'])
# print (iris['DESCR'])

