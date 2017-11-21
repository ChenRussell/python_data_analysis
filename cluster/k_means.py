# -*- coding:utf-8 -*-
# k-means实验

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import sys,getopt
import sys
import pandas as pd


# opts, args = getopt.getopt(sys.argv[1:], "hi:1:2:")
# for op, value in opts:
#     if op == "-i":
#         file_data = value
#     elif op == "-h":
#         k_value = value

file_data = sys.argv[1]
output_data = sys.argv[2]
k_value = sys.argv[3]

# plt.figure(figsize=(12, 12))
plt.figure()

# 选取样本数量
n_samples = 1500
# 选取随机因子
random_state = 170
# 获取数据集

# z = np.loadtxt(file_data, delimiter=',')
df = pd.read_csv(file_data, delimiter=',', skiprows=0)
print(df)
z = df.as_matrix()

print(z)
print(len(z))

y_pred = KMeans(n_clusters=int(k_value), random_state=random_state).fit_predict(z)
print('y_pred', y_pred, len(y_pred))
df['cluster']=y_pred
print(df)
df.to_csv(output_data, index=False, header=False)
# df.to_csv(output_data, index=False, columns=['literature', 'math', 'cluster'])
colors = ['r', 'b', 'g', 'm', 'y', 'orange', 'c', 'k']
markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p']
for i in range(int(k_value)):
    plt.scatter(z[y_pred==i][:, 0], z[y_pred==i][:, 1], marker=markers[i],color=colors[i])
# plt.scatter(z[y_pred==1][:, 0], z[y_pred==1][:, 1], marker='+',color='r')
# plt.scatter(z[y_pred==2][:, 0], z[y_pred==2][:, 1], marker='1',color='g')
# plt.scatter(z[y_pred==3][:, 0], z[y_pred==3][:, 1], marker='2',color='m')
# plt.scatter(z[y_pred==4][:, 0], z[y_pred==4][:, 1], marker='o',color='orange')
# plt.scatter(z[y_pred==5][:, 0], z[y_pred==5][:, 1], marker='3',color='y')
plt.title("k-means cluster of "+file_data)
# plt.show()
plt.savefig(file_data+'_'+k_value+'clusters.png')