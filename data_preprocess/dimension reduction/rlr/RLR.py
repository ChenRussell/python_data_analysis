#-*- coding: utf-8 -*-
#逻辑回归 自动建模
import pandas as pd
import csv

#参数初始化
filename = 'bankloan.xls'
data = pd.read_excel(filename)
print(data)

x = data.iloc[:,:8].as_matrix()
y = data.iloc[:,8].as_matrix()

from sklearn.linear_model import RandomizedLogisticRegression as RLR
rlr = RLR(selection_threshold=0.25) #建立随机逻辑回归模型，筛选变量
rlr.fit(x, y) #训练模型
rlr.get_support() #获取特征筛选结果，也可以通过.scores_方法获取各个特征的分数
print(rlr.get_support())
print(u'通过随机逻辑回归模型筛选特征结束。')
print(u'有效特征为：%s' % ','.join(data.columns[rlr.get_support()]))
x = data[data.columns[rlr.get_support()]].as_matrix() #筛选好特征
print(type(x))
df = pd.DataFrame(x)
df[4] = y
print(df)
df.to_csv('out_rlr_df2.csv', index=False, header=False, float_format = '%.2f')
# f = open('out_rlr.csv', 'w')
# writer = csv.writer(f, delimiter=',', lineterminator='\n')
# for i in x:
#     writer.writerow(i)
