from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
import pandas as pd

inputfile = 'result.csv'
# outputfile = '../tmp/dimention_reducted.xls' #降维后的数据

data = pd.read_csv(inputfile, header = None) #读入数据

# model = SelectKBest(chi2,k=3)
# print(model.fit_transform(data))    # 先调用fit(x,y)方法。在调用transform(x)方法

model = VarianceThreshold(threshold=3)
res = model.fit_transform(data)
print(res.shape)
print(model.variances_)
vari = model.variances_
print(type(vari))
sum = 0
for i in vari:
    sum += i
print(sum)

print(list(vari))
arr = []
for i in vari:
    # arr.append(round(i/sum,4))
    arr.append('%.5f' % (i/sum) )
print(arr)
# print(model.get_support())