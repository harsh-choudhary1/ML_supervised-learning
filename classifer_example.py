# using iris dataset ploting the graph to presenting the  data value of verginika flower in iris dataset
 
from sklearn.datasets import load_iris 
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

data = load_iris()

print(data.keys())
x = data["data"][ : , 3:]
y =( data["target"] == 2 ).astype(np.int_)
print(x,y)

clf = LogisticRegression()
clf.fit(x,y)

xx = np.linspace(0,3,10).reshape(-1,1)
# print(xx)
y_p = clf.predict_proba(xx)
plt.plot(xx,y_p[:,1] , "g-", label="verginika")
plt.show()
