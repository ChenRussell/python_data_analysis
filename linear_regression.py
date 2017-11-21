# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import pandas as pd

boston = load_boston()
print(boston.keys())
# result:
# ['data', 'feature_names', 'DESCR', 'target']

print(boston.feature_names)
# print(boston)
# result:
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

# print boston.DESCR    # 取消注释并运行，可查看数据说明文档
x = boston.data[:, np.newaxis, 5]
print(np.newaxis)
# print(x)
# print(x[0])
# x = boston.data
print(x)
print(type(x))
# print(boston.data[:, 5])
# print(type(boston.data[:, 5]))
y = boston.target
# print(y)
# print(type(y))
# data = pd.read_csv('data/BostonHousePricing.csv', header=None)
# data['result'] = y
# print(data)
# col = boston.feature_names
# print(col)
# col = np.append(col, 'result')
# print(col)
# data.columns = col
# data.to_csv('data/boston_label.csv', index=False, float_format='%.2f')
# print(data)
# print(x,len(x))
# print(y,len(y))
lm = LinearRegression()    # 声明并初始化一个线性回归模型的对象
lm.fit(x, y)    # 拟合模型，或称为训练模型
print(u'方程的确定性系数(R^2): %.2f' % lm.score(x, y))
# result: 方程的确定性系数(R^2): 0.48

plt.scatter(x, y, color='green')    # 显示数据点
plt.plot(x, lm.predict(x), color='blue', linewidth=3)    # 画出回归直线
plt.xlabel('Average Number of Rooms per Dwelling (RM)')
plt.ylabel('Housing Price')
plt.title('2D Demo of Linear Regression')
plt.show()
