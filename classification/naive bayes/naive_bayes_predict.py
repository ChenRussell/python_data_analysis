import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
import getopt
import sys

shortargs = 'i:o:hm:'   # 短选项模式：冒号表示该选项必须有附加的参数，不带冒号表示该选项不附带参数
longargs = ['ifile=', 'ofile=']     # 长选项模式：=表示如果设置该选项，必须有附加的参数，否则就不附加参数
inputfile = ''
outputfile = ''
modelfile = ''


def predict():
    dat = pd.read_csv(inputfile)
    print(dat)
    x = dat
    model = joblib.load(modelfile)
    dat['label'] = model.predict(x)
    dat.to_csv(outputfile, index=False, float_format='%.2f', encoding='utf-8')


try:
    opts, args = getopt.getopt(sys.argv[1:], shortargs, longargs)
except getopt.GetoptError:
    print('naive_bayes_predict.py -i <inputfile> -o <outputfile>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('naive_bayes_predict.py -i <inputfile> -o <outputfile>')
        sys.exit()
    elif opt in ("-i", "--ifile"):
        inputfile = arg
    elif opt in ("-o", "--ofile"):
        outputfile = arg
    elif opt == '-m':
        modelfile = arg

predict()