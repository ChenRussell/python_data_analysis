from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import pandas as pd
import getopt
import sys

shortargs = 'i:o:k:'   # 短选项模式：冒号表示该选项必须有附加的参数，不带冒号表示该选项不附带参数
longargs = ['ifile=', 'o:file']     # 长选项模式：=表示如果设置该选项，必须有附加的参数，否则就不附加参数
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
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=int(k), step=1)
    rfe.fit(x, y)
    x_new = data[data.columns[rfe.get_support()]]
    x_new['label'] = y
    x_new.to_csv(outputfile, index=False, float_format='%.2f', encoding='utf-8')    # 如果dataframe是原有的截出来的，就不会有00的两位小数位，如果是新定义的，to_csv就会有00两位小数位


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
    elif opt == '-k':
        k = arg
train_model()
