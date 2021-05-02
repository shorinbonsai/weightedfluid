import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataframe = pd.read_csv("07_15transpose.csv", error_bad_lines=False)
dataframe2 = pd.read_csv("/home/james/4F90/weightedfluid/data_07_15/raw.csv", error_bad_lines=False)
print(dataframe2.head())
print(dataframe2.isnull().values.any())

df1 = dataframe.iloc[:,:7]

# alg1cont5 = dataframe.Algo1fluidCont5
sns.violinplot(data=df1)

plt.savefig('control.png')
plt.show()
plt.close()

sns.boxplot(data=df1)
plt.savefig('controlbox.png')
plt.show()
plt.close()
