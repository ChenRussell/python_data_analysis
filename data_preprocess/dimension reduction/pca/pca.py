import pandas as pd
from sklearn.decomposition import PCA
import getopt
import sys

shortargs = 'i:o:k:'   # 短选项模式：冒号表示该选项必须有附加的参数，不带冒号表示该选项不附带参数
longargs = ['ifile=', 'o:file']     # 长选项模式：=表示如果设置该选项，必须有附加的参数，否则就不附加参数
inputfile = ''
outputfile = ''
k = 2


def train_model():
    data = pd.read_csv(inputfile)   # 程序中会带有属性名
    x = data
    pca = PCA(int(k))
    pca.fit(x)
    x_new = pca.transform(data)
    print(x_new)
    x_new = pd.DataFrame(x_new)
    x_new.to_csv(outputfile, index=False, header=None, float_format='%.2f', encoding='utf-8')    #pd.read_csv用utf-8读文件，pd.to_csv()却用gbk存，日了狗



try:
    opts, args = getopt.getopt(sys.argv[1:], shortargs, longargs)
except getopt.GetoptError:
    print('pca.py -i <inputfile> -o <outputfile>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('pca.py -i <inputfile> -o <outputfile>')
        sys.exit()
    elif opt in ("-i", "--ifile"):
        inputfile = arg
    elif opt in ("-o", "--ofile"):
        outputfile = arg
    elif opt == '-k':
        k = arg
train_model()