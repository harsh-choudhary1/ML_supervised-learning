import matplotlib as plt
import numpy as np 
from sklearn import datasets, linear_model

datas = datasets.load_iris()
# print(data_s.keys())
#output of data_s.keys# keys = = dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
datas_x = iris.data[:, np.newaxis,2]
print(datas_x)
