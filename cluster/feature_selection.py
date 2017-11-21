import pandas as pd
import sys


input_data = sys.argv[1]
output_data = sys.argv[2]


df = pd.read_csv(input_data)
# df = pd.DataFrame(df.as_matrix())
df = df[[0, 1]]
print(df)

df.to_csv(output_data, index=False)