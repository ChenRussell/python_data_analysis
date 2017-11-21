import pandas as pd
from sklearn import svm
from sklearn.externals import joblib
import getopt
import sys

shortargs = 'i:o:hm:'   # 短选项模式：冒号表示该选项必须有附加的参数，不带冒号表示该选项不附带参数
longargs = ['ifile=', 'ofile=']     # 长选项模式：=表示如果设置该选项，必须有附加的参数，否则就不附加参数
inputfile = ''
outputfile = ''
modelfile = ''


def predict():
    dat = pd.read_csv(inputfile)   # 如果数据没有属性名，则加上header=None;否则不加
    print(dat)
    x = dat
    # x = dat.iloc[:, :-1]
    # y = dat.iloc[:, -1]
    model = joblib.load(modelfile)
    dat['label'] = model.predict(x)
    dat.to_csv(outputfile, index=False, float_format='%.2f', encoding='utf-8')  # 输出文件保留属性名,u8编码


try:
    opts, args = getopt.getopt(sys.argv[1:], shortargs, longargs)
except getopt.GetoptError:
    print('svm_predict.py -i <inputfile> -o <outputfile>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('svm_predict.py -i <inputfile> -o <outputfile>')
        sys.exit()
    elif opt in ("-i", "--ifile"):
        inputfile = arg
    elif opt in ("-o", "--ofile"):
        outputfile = arg
    elif opt == '-m':
        modelfile = arg

predict()