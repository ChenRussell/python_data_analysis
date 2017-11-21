import pandas as pd

df = pd.read_csv('data_analysis.csv',header=None)
# print(df)
a = df.iloc[13, 1]
print(a)
print(type(a))
s = df[1]
df = df.replace("\\N", 111)
print(df)
# s.replace('\\N',0)
# print(s)
# s[s=='\\N'] = 0
# print(s)
# df[df == '\\N'] = 0
# df[df[1]=='\\N'] = 0
# print(df)