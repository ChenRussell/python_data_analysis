# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.datasets import load_iris
import pandas as pd
# import mlpy
from sklearn import svm

# iris = load_iris()     # 加载数据
# iris = np.genfromtxt('Narcissus.txt',delimiter=' ', skip_header=1)
# iris = np.loadtxt('../data/Narcissus.txt', delimiter='\t', skiprows=1)
# print(iris.data)
# print(len(iris.data))
# iris = pd.read_csv('Narcissus.txt', delimiter=' ', skiprows=0).as_matrix()
# print(iris)
# X = iris[:, :2]    # 为方便画图，仅采用数据的其中两个特征
# y = iris.target
# y = iris[:, 4]

data = pd.read_csv('electricity_steal_train.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X = X.as_matrix()
y = y.as_matrix()

print(X, len(X))
print(y, len(y))
# print(iris.DESCR)
# print(iris.feature_names)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# clf = KNeighborsClassifier(n_neighbors=15, weights='uniform')    # 初始化分类器对象
# clf.fit(X, y)
# svm = mlpy.LibSvm(kernel_type='linear', gamma=20 )
# svm.learn(X, y)
model = svm.SVC()
model.fit(X, y)
print(model.score(X, y))

# 画出决策边界，用不同颜色表示
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
print('Z:', Z, len(Z))
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)    # 绘制预测结果图

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)    # 补充训练数据点
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class SVM classification, predict accuracy:%.2f" % model.score(X, y))
plt.savefig('image/SVM.png')
plt.show()