from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import getopt
import sys

shortargs = 'i:o:k:'   # 短选项模式：冒号表示该选项必须有附加的参数，不带冒号表示该选项不附带参数
longargs = ['ifile=', 'o:file']     # 长选项模式：=表示如果设置该选项，必须有附加的参数，否则就不附加参数
inputfile = ''
outputfile = ''
k = 2
# iris = load_iris()
# x, y = iris.data, iris.target
# print(x)


def train_model():
    data = pd.read_csv(inputfile)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    # x_new = SelectKBest(chi2, k=int(k)).fit_transform(x, y)
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x, y)    # 参数C控制稀疏性：C越小，被选中的特征越少
    model = SelectFromModel(lsvc, prefit=True)
    x_new = model.transform(x)
    print(x_new)
    out_data = pd.DataFrame(x_new)
    out_data['t'] = y
    print(out_data)
    out_data.to_csv(outputfile, index=False, header=False, float_format='%.2f')



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
