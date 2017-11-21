# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import pandas as pd

# iris = load_iris()     # 加载数据
# iris = np.genfromtxt('Narcissus.txt',delimiter=' ', skip_header=1)
iris = np.loadtxt('data/Narcissus.txt', delimiter='\t', skiprows=1)
# iris = np.loadtxt('wine.data', delimiter=',')
# print(iris.data)
# print(len(iris.data))
# iris = pd.read_csv('Narcissus.txt', delimiter=' ', skiprows=0).as_matrix()
print(iris)
X = iris[:, :2]    # 为方便画图，仅采用数据的其中两个特征
# X = iris[:, 1:3]    # 为方便画图，仅采用数据的其中两个特征
# y = iris.target
y = iris[:, 4]
# y = iris[:, 0]
print(X, len(X))
print(y, len(y))
# print(iris.DESCR)
# print(iris.feature_names)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

clf = KNeighborsClassifier(n_neighbors=15, weights='uniform')    # 初始化分类器对象
clf.fit(X, y)

from cm_plot import * #导入自行编写的混淆矩阵可视化函数
cm_plot(y, clf.predict(X)).show() #显示混淆矩阵可视化结果

# 画出决策边界，用不同颜色表示
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)    # 绘制预测结果图

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)    # 补充训练数据点
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("kNN分类-3, 预测精度为:%.2f" % clf.score(X, y))
plt.savefig('image/knn.png')
plt.show()