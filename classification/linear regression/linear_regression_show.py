# -*- coding:utf-8 -*-
# k-means实验

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys,getopt
import sys
import pandas as pd


shortargs = 'i:o:k:p:'   # 短选项模式：冒号表示该选项必须有附加的参数，不带冒号表示该选项不附带参数
longargs = ['ifile=', 'ofile=']     # 长选项模式：=表示如果设置该选项，必须有附加的参数，否则就不附加参数
inputfile = ''
outputfile = ''
inputfile2 =''
k = 2

try:
    opts, args = getopt.getopt(sys.argv[1:], shortargs, longargs)
except getopt.GetoptError:
    print('cluster_show.py -i <inputfile> -o <outputfile>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('cluster_show.py -i <inputfile> -o <outputfile>')
        sys.exit()
    elif opt in ("-i", "--ifile"):
        inputfile = arg
    elif opt in ("-o", "--ofile"):
        outputfile = arg
    elif opt == '-p':
        inputfile2 = arg


data1 = pd.read_csv(inputfile)
data2 = pd.read_csv(inputfile2)
x = data1.iloc[:, :-1]
y1 = data1.iloc[:, -1]
y2 = data2.iloc[:, -1]
plt.scatter(x, y1, color='green')    # 显示数据点
plt.plot(x, y2, color='blue', linewidth=3)    # 画出回归直线
plt.xlabel('Average Number of Rooms per Dwelling (RM)')
plt.ylabel('Housing Price')
plt.title('2D Demo of Linear Regression')
# plt.show()
plt.savefig(outputfile)