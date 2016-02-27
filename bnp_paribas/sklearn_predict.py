"""
Classification models using sklearn
"""
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

# avoid scientic notation when printing matrices
np.set_printoptions(suppress=True)

train = pd.read_csv(
    './bnp_paribas/data/train.csv',
    index_col=0)

# plot feature histograms

fig = plt.figure()
ax = fig.add_subplot(111)
var = 'v129'
ax.hist(train[var], bins=10, range=(train[var].min(), train[var].max()))
plt.show()

train.describe()

# fill missing data

train_zero = train.fillna(0)
train_mean = train.fillna(train.mean())

# model

sgd = SGDClassifier(
    loss="log", penalty="elasticnet",
    n_iter=100, random_state=123)

sgd.fit(train_zero.iloc[:, 1:], train_zero.iloc[:, 0])
print sgd.score(train_zero.iloc[:, 1:], train_zero.iloc[:, 0])

# calculate log loss
train_labels = sgd.predict_proba(train_zero.iloc[:, 1:])
print log_loss(train_zero.iloc[:, 0], train_labels)

# write out predictions

with open('./bnp_paribas/data/predictions.csv', 'w') as predictions_file:
    csv_writer = csv.writer(predictions_file)
    csv_writer.writerow(["ID", "PredictedProb"])
    csv_writer.writerows(zip(train_zero.index, train_labels[:, 0]))

# random observations
"""
In [35]: df.groupby('v3').count()
Out[35]:
    target     v1     v2     v4     v5     v6     v7     v8     v9     v10  \
v3
A      227    147    147    147    150    147    147    150    147     227
B       53     26     26     26     27     26     26     27     26      53
C   110584  61543  61578  61578  62660  61543  61543  62665  61525  110501

---> about half the data has all values

In [55]: for column in df.columns:
   ....:         if str(df.dtypes[column]) == 'category':
   ....:                 print column, len(df[column].cat.categories)
   ....:
v3 3
v22 18210
v24 5
v30 7
v31 3
v47 10
v52 12
v56 122
v66 3
v71 9
v74 3
v75 4
v79 18
v91 7
v107 7
v110 3
v112 22
v113 36
v125 90

In [54]: df[['v3', 'target']].groupby('v3').mean()
Out[54]:
      target
v3
A   0.955947   <--- very likely
B   0.886792
C   0.759396

In [56]: df[['v24', 'target']].groupby('v24').mean()
Out[56]:
       target
v24
A    0.732119   <--- not very important
B    0.763313
C    0.799588
D    0.744389
E    0.756384

---> use delta from df['target'].mean() == 0.761 to decide if to make a feature?

In [98]: df_cats.describe()
Out[98]:
            v3     v24    v30     v31     v47     v52     v56     v66     v71  \
count   110864  114321  54211  110864  114321  114318  107439  114321  114321
unique       3       5      7       3      10      12     122       3       9
top          C       E      C       A       C       J      BW       A       F
freq    110584   55177  32178   88347   55425   11103   11351   70353   75094

           v74     v75     v79     v91    v107    v110    v112   v113    v125
count   114321  114321  114321  114318  114318  114321  113939  59017  114244
unique       3       4      18       7       7       3      22     36      90
top          B       D       C       A       E       A       F      G      BM
freq    113560   75087   34561   27079   27079   55688   21671  16252    5759

cluster into groups based on data availability
"""
