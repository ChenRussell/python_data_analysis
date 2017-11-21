import pandas as pd
from sklearn.linear_model import RandomizedLogisticRegression as RLR
from sklearn.externals import joblib
import getopt
import sys

shortargs = 'i:o:htp'   # 短选项模式：冒号表示该选项必须有附加的参数，不带冒号表示该选项不附带参数
longargs = ['ifile=', 'o:file']     # 长选项模式：=表示如果设置该选项，必须有附加的参数，否则就不附加参数
inputfile = ''
outputfile = ''
flag = 0


def train_model():
    data = pd.read_excel(inputfile)
    print(data)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    model = RLR(selection_threshold=0.25) #建立随机逻辑回归模型，筛选变量
    model.fit(x, y)
    # x = data[data.columns[model.get_support()]].as_matrix()  # 筛选好特征
    # print(type(x))
    # df = pd.DataFrame(x)
    # df[4] = y
    # print(df)
    x = data[data.columns[model.get_support()]]
    x['a'] = y
    x.to_csv(outputfile, index=False, header=False, float_format='%.2f')



try:
    opts, args = getopt.getopt(sys.argv[1:], shortargs, longargs)
except getopt.GetoptError:
    print('svm.py -i <inputfile> -o <outputfile>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('svm.py -i <inputfile> -o <outputfile>')
        sys.exit()
    elif opt in ("-i", "--ifile"):
        inputfile = arg
    elif opt in ("-o", "--ofile"):
        outputfile = arg

train_model()