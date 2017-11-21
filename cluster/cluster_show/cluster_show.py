# -*- coding:utf-8 -*-
# k-means实验

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys,getopt
import sys
import pandas as pd


shortargs = 'i:o:k:'   # 短选项模式：冒号表示该选项必须有附加的参数，不带冒号表示该选项不附带参数
longargs = ['ifile=', 'ofile=']     # 长选项模式：=表示如果设置该选项，必须有附加的参数，否则就不附加参数
inputfile = ''
outputfile = ''
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
    elif opt == '-k':
        k = arg

plt.figure()

df = pd.read_csv(inputfile)
z = df.as_matrix()
label = z[:, -1]    # last column show be label column
colors = ['r', 'b', 'g', 'm', 'y', 'orange', 'c', 'k']
markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p']
for i in range(int(k)):
    plt.scatter(z[label == i][:, 0], z[label == i][:, 1], marker=markers[i],color=colors[i])
plt.title("k-means cluster of "+outputfile)
# plt.show()
plt.savefig(outputfile)