from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import getopt
import sys

shortargs = 'i:o:k:'   # 短选项模式：冒号表示该选项必须有附加的参数，不带冒号表示该选项不附带参数
longargs = ['ifile=', 'ofile=']     # 长选项模式：=表示如果设置该选项，必须有附加的参数，否则就不附加参数
inputfile = ''
outputfile = ''
k = 2


def train_model():
    data = pd.read_csv(inputfile)
    # x_new = SelectKBest(chi2, k=int(k)).fit_transform(x, y)
    # result = KMeans(n_clusters=int(k), random_state=random_state).fit_predict(data)
    result = AgglomerativeClustering(affinity='euclidean',linkage='ward',n_clusters=2).fit_predict(data)
    data['result'] = result
    data.to_csv(outputfile, index=False, float_format='%.2f', encoding='utf-8')



try:
    opts, args = getopt.getopt(sys.argv[1:], shortargs, longargs)
except getopt.GetoptError:
    print('AgglomerativeClustering.py -i <inputfile> -o <outputfile>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('AgglomerativeClustering.py -i <inputfile> -o <outputfile>')
        sys.exit()
    elif opt in ("-i", "--ifile"):
        inputfile = arg
    elif opt in ("-o", "--ofile"):
        outputfile = arg
    elif opt == '-k':
        k = arg
train_model()