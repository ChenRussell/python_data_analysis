# -*- coding:utf-8 -*-
# 使用ID3算法进行分类
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC, export_graphviz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = pd.read_csv('titanic_data.csv', encoding='utf-8')
data.drop(['PassengerId'], axis=1, inplace=True)    # 舍弃ID列，不适合作为特征

# 数据是类别标签，将其转换为数，用1表示男，0表示女。
data.loc[data['Sex'] == 'male', 'Sex'] = 1
data.loc[data['Sex'] == 'female', 'Sex'] = 0
data.fillna(int(data.Age.mean()), inplace=True)
print(data.head(5))   # 查看数据

X = data.iloc[:, 1:3]    # 为便于展示，未考虑年龄（最后一列）
y = data.iloc[:, 0]

iris = np.loadtxt('Narcissus.txt', delimiter='\t', skiprows=1)
X = iris[:, :2]    # 为方便画图，仅采用数据的其中两个特征
y = iris[:, 4]

dtc = DTC(criterion='entropy')    # 初始化决策树对象，基于信息熵
dtc.fit(X, y)    # 训练模型
print('输出准确率：', dtc.score(X,y))
# print(dtc.predict(X))
# print(X.as_matrix())
# print(y.as_matrix())
# X=X.as_matrix()
# y=y.as_matrix()
# # 可视化决策树，导出结果是一个dot文件，需要安装Graphviz才能转换为.pdf或.png格式
# with open('../tmp/tree.dot', 'w') as f:
#     f = export_graphviz(dtc, feature_names=X.columns, out_file=f)
# 画出决策边界，用不同颜色表示
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

Z = dtc.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)


plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)    # 绘制预测结果图

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)    # 补充训练数据点
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class decision_tree classification, predict accuracy:%.2f" % dtc.score(X, y))
plt.savefig('image/decision_tree.png')
plt.show()