import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

SMALL_SIZE = 10
EXTRA_SMALL = 5
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)

dataframe = pd.read_csv("07_15transpose.csv", error_bad_lines=False)
dataframe2 = pd.read_csv("/home/james/4F90/weightedfluid/05_06transpose.csv", error_bad_lines=False)
dataframe3 = pd.read_csv("/home/james/4F90/weightedfluid/modularityTranspose.csv", error_bad_lines=False)
print(dataframe3.head())
print(dataframe3.isnull().values.any())

df = dataframe3.iloc[:,:28]
df1 = dataframe3.iloc[:,29:]

sns.boxplot(data=df)
plt.savefig('modular_5_6.png')
plt.show()
plt.close()

sns.boxplot(data=df1)
plt.savefig('modular_7_15.png')
plt.show()
plt.close()

# sns.violinplot(data=alg1C)
# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("07/15 Control Comparison")

# plt.savefig('Alg1CV_7_15.png')
# plt.show()
# plt.close()

# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("07/15 Control Comparison")
# sns.boxplot(data=alg1C)
# plt.savefig('Alg1CB_7_15.png')
# plt.show()
# plt.close()

# alg1L = dataframe.iloc[:,7:14]

# sns.violinplot(data=alg1L)
# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("07/15 Louvain Comparison")

# plt.savefig('Alg1LV_7_15.png')
# plt.show()
# plt.close()

# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("07/15 Louvain Comparison")
# sns.boxplot(data=alg1L)
# plt.savefig('Alg1LB_7_15.png')
# plt.show()
# plt.close()

# alg1LR = dataframe.iloc[:,14:21]

# sns.violinplot(data=alg1LR)
# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("07/15 Louvain with Random Comparison")

# plt.savefig('Alg1LRV_7_15.png')
# plt.show()
# plt.close()

# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("07/15 Louvain with Random Comparison")
# sns.boxplot(data=alg1LR)
# plt.savefig('Alg1LRB_7_15.png')
# plt.show()
# plt.close()

# matplotlib.rc('font', size=EXTRA_SMALL)

# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("07/15 Overall Algorithm 1")

# sns.boxplot(data=df1)
# plt.savefig('Alg1Overall_7_15.png')
# plt.show()
# plt.close()

# #############################################################################
# matplotlib.rc('font', size=SMALL_SIZE)
# matplotlib.rc('axes', titlesize=SMALL_SIZE)
# alg2C = dataframe.iloc[:,21:28]
# df2 = dataframe.iloc[:,21:42]

# sns.violinplot(data=alg2C)
# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("07/15 Control Comparison")

# plt.savefig('Alg2CV_7_15.png')
# plt.show()
# plt.close()

# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("07/15 Control Comparison")
# sns.boxplot(data=alg2C)
# plt.savefig('Alg2CB_7_15.png')
# plt.show()
# plt.close()

# alg2L = dataframe.iloc[:,28:35]

# sns.violinplot(data=alg2L)
# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("07/15 Louvain Comparison")

# plt.savefig('Alg2LV_7_15.png')
# plt.show()
# plt.close()

# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("07/15 Louvain Comparison")
# sns.boxplot(data=alg2L)
# plt.savefig('Alg2LB_7_15.png')
# plt.show()
# plt.close()

# alg2LR = dataframe.iloc[:,35:42]

# sns.violinplot(data=alg2LR)
# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("07/15 Louvain with Random Comparison")

# plt.savefig('Alg2LRV_7_15.png')
# plt.show()
# plt.close()

# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("07/15 Louvain with Random Comparison")
# sns.boxplot(data=alg2LR)
# plt.savefig('Alg2LRB_7_15.png')
# plt.show()
# plt.close()

# matplotlib.rc('font', size=EXTRA_SMALL)

# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("07/15 Overall Algorithm 2")


# sns.boxplot(data=df2)
# plt.savefig('Alg2Overall_7_15.png')
# plt.show()
# plt.close()

# ########################### 05/06 ###########################################
# matplotlib.rc('font', size=SMALL_SIZE)
# matplotlib.rc('axes', titlesize=SMALL_SIZE)

# alg1C = dataframe2.iloc[:,:7]
# df1 = dataframe2.iloc[:,:21]

# sns.violinplot(data=alg1C)
# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("05/06 Control Comparison")

# plt.savefig('Alg1CV_5_6.png')
# plt.show()
# plt.close()

# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("05/06 Control Comparison")
# sns.boxplot(data=alg1C)
# plt.savefig('Alg1CB_5_6.png')
# plt.show()
# plt.close()

# alg1L = dataframe2.iloc[:,7:14]

# sns.violinplot(data=alg1L)
# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("05/06 Louvain Comparison")

# plt.savefig('Alg1LV_5_6.png')
# plt.show()
# plt.close()

# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("05/06 Louvain Comparison")
# sns.boxplot(data=alg1L)
# plt.savefig('Alg1LB_5_6.png')
# plt.show()
# plt.close()

# alg1LR = dataframe2.iloc[:,14:21]

# sns.violinplot(data=alg1LR)
# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("05/06 Louvain with Random Comparison")

# plt.savefig('Alg1LRV_5_6.png')
# plt.show()
# plt.close()

# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("05/06 Louvain with Random Comparison")
# sns.boxplot(data=alg1LR)
# plt.savefig('Alg1LRB_5_6.png')
# plt.show()
# plt.close()

# matplotlib.rc('font', size=EXTRA_SMALL)

# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("05/06 Overall Algorithm 1")

# sns.boxplot(data=df1)
# plt.savefig('Alg1Overall_5_6.png')
# plt.show()
# plt.close()

# #############################################################################
# matplotlib.rc('font', size=SMALL_SIZE)
# matplotlib.rc('axes', titlesize=SMALL_SIZE)
# alg2C = dataframe2.iloc[:,21:28]
# df2 = dataframe2.iloc[:,21:42]

# sns.violinplot(data=alg2C)
# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("05/06 Control Comparison")

# plt.savefig('Alg2CV_5_6.png')
# plt.show()
# plt.close()

# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("05/06 Control Comparison")
# sns.boxplot(data=alg2C)
# plt.savefig('Alg2CB_5_6.png')
# plt.show()
# plt.close()

# alg2L = dataframe2.iloc[:,28:35]

# sns.violinplot(data=alg2L)
# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("05/06 Louvain Comparison")

# plt.savefig('Alg2LV_5_6.png')
# plt.show()
# plt.close()

# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("05/06 Louvain Comparison")
# sns.boxplot(data=alg2L)
# plt.savefig('Alg2LB_5_6.png')
# plt.show()
# plt.close()

# alg2LR = dataframe2.iloc[:,35:42]

# sns.violinplot(data=alg2LR)
# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("05/06 Louvain with Random Comparison")

# plt.savefig('Alg2LRV_5_6.png')
# plt.show()
# plt.close()

# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("05/06 Louvain with Random Comparison")
# sns.boxplot(data=alg2LR)
# plt.savefig('Alg2LRB_5_6.png')
# plt.show()
# plt.close()

# matplotlib.rc('font', size=EXTRA_SMALL)

# plt.xlabel("Algorithms and Community Numbers")
# plt.ylabel("Adjusted Rand Index")
# plt.title("05/06 Overall Algorithm 2")


# sns.boxplot(data=df2)
# plt.savefig('Alg2Overall_5_6.png')
# plt.show()
# plt.close()