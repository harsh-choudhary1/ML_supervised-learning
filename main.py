# loading modules 
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# colecting data from iris dataset
iris = datasets.load_iris()
# print(iris.DESCR)
features  = iris.data
lables =  iris.target
# print (features[0],lables[0])
# classifying the data 

clf =  KNeighborsClassifier()
clf.fit(features,lables)

preds = clf.predict([[17,24,41,51]])
print(preds)

