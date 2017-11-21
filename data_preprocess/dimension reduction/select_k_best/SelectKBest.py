from sklearn.datasets import  load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
import getopt
import sys

shortargs = 'i:o:k:'   # 短选项模式：冒号表示该选项必须有附加的参数，不带冒号表示该选项不附带参数
longargs = ['ifile=', 'ofile=']     # 长选项模式：=表示如果设置该选项，必须有附加的参数，否则就不附加参数
inputfile = ''
outputfile = ''
k = 4
# iris = load_iris()
# x, y = iris.data, iris.target
# print(x)


def train_model():
    data = pd.read_csv(inputfile)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    # x_new = SelectKBest(chi2, k=int(k)).fit_transform(x, y)
    model = SelectKBest(chi2, k=int(k))
    model.fit(x, y)
    x_new = data[data.columns[model.get_support()]]
    print(x_new)
    # out_data = pd.DataFrame(x_new)
    x_new['label'] = y
    print(x_new)
    x_new.to_csv(outputfile, index=False, float_format='%.2f', encoding='utf-8')    #pd.read_csv用utf-8读文件，pd.to_csv()却用gbk存，日了狗



try:
    opts, args = getopt.getopt(sys.argv[1:], shortargs, longargs)
except getopt.GetoptError:
    print('SelectKBest.py -i <inputfile> -o <outputfile>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('SelectKBest.py -i <inputfile> -o <outputfile>')
        sys.exit()
    elif opt in ("-i", "--ifile"):
        inputfile = arg
    elif opt in ("-o", "--ofile"):
        outputfile = arg
    elif opt == '-k':
        k = arg
train_model()