import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import getopt
import sys

shortargs = 'i:o:htp'   # 短选项模式：冒号表示该选项必须有附加的参数，不带冒号表示该选项不附带参数
longargs = ['ifile=', 'o:file']     # 长选项模式：=表示如果设置该选项，必须有附加的参数，否则就不附加参数
inputfile = ''
outputfile = ''
flag = 0


def train_model():
    dat = pd.read_csv(inputfile, header=None)
    x = dat.iloc[:, :-1]
    y = dat.iloc[:, -1]
    model = KNeighborsClassifier(n_neighbors=15, weights='uniform')
    model.fit(x, y)
    joblib.dump(model, "knn.m")


def predict():
    dat = pd.read_csv(inputfile, header=None)
    print(dat)
    # x = dat
    x = dat.iloc[:, :-1]
    y = dat.iloc[:, -1]
    model = joblib.load("knn.m")
    dat['a'] = model.predict(x)
    dat.to_csv(outputfile, index=False, header=False, float_format='%.2f')
    print(u'模型的平均正确率为：%s' % model.score(x, y))


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
    elif opt == '-t':
        flag = 1
    elif opt == '-p':
        flag = 2

if flag == 1:
    train_model()
if flag == 2:
    predict()