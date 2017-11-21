import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame([1,2,3],[4,5,6])
print(df)
print(df.to_dict())
df.plot()

# plt.show()