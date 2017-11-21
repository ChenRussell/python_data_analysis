import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.externals import joblib
import getopt
import sys

shortargs = 'i:o:h'   # 短选项模式：冒号表示该选项必须有附加的参数，不带冒号表示该选项不附带参数
longargs = ['ifile=', 'o:file']     # 长选项模式：=表示如果设置该选项，必须有附加的参数，否则就不附加参数
inputfile = ''
outputfile = ''


def train_model():
    dat = pd.read_csv(inputfile)   # 如果数据没有属性名就加上header=None;否则不加
    x = dat.iloc[:, :-1]
    y = dat.iloc[:, -1]
    model = DTC(criterion='entropy')
    model.fit(x, y)
    joblib.dump(model, outputfile)


try:
    opts, args = getopt.getopt(sys.argv[1:], shortargs, longargs)
except getopt.GetoptError:
    print('dt_train.py -i <inputfile> -o <outputfile>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('dt_train.py -i <inputfile> -o <outputfile>')
        sys.exit()
    elif opt in ("-i", "--ifile"):
        inputfile = arg
    elif opt in ("-o", "--ofile"):
        outputfile = arg

train_model()