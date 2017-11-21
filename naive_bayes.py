from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

data = pd.read_csv('Narcissus.txt', skiprows=0, delimiter='\t').as_matrix()
print(data)
clf = GaussianNB()
X = data[:, :2]
y = data[:, 4]
clf.fit(X, y)
print(clf.score(X, y))
y_pred = clf.predict(X)
print("Number of mislabeled points out of a total %d points: %d" % (data.shape[0], (data[:, 4]!= y_pred).sum()))

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)    # 绘制预测结果图

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)    # 补充训练数据点
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class naive_bayes classification, predict accuracy:%.2f" % clf.score(X, y))
plt.savefig('image/naive_bayes.png')
plt.show()