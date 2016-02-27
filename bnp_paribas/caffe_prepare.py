"""
Run this script to prepare the train.csv data for analysis in Caffe.

Afterwards, run this command to train the model:
    ../caffe/build/tools/caffe train -solver bnp_paribas/data/caffe/nonlinear_solver.prototxt

Based on http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/02-brewing-logreg.ipynb
"""
import os
import h5py
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split

# avoid scientic notation when printing matrices
np.set_printoptions(suppress=True)

# load data from CSV

train = pd.read_csv(
    './bnp_paribas/data/train.csv',
    index_col=0)

test = pd.read_csv(
    './bnp_paribas/data/test.csv',
    index_col=0)

# combine the datasets to apply the same feature engineering to both
all_data = pd.concat((train, test))
max_categories = 100

# convert categorical columns into a series of binary dummy columns
for column in all_data.columns:
    if str(all_data.dtypes[column]) == 'object':

        if len(all_data[column].unique()) > max_categories:
            # limit the number of dummies to `max_categories`
            to_keep = list(all_data[column].value_counts()[:max_categories])
            to_keep.append(np.nan)
            # use a unique category for overflow, not NaN
            all_data.loc[:, column] = all_data[column].apply(lambda x: x if x in to_keep else 'other')

        all_data.loc[:, column] = all_data[column].astype('category')
        dummies = pd.get_dummies(all_data[column], prefix=column, dummy_na=True)
        all_data = pd.concat((all_data, dummies), axis=1)
        del all_data[column]

# replace NaN with a number
# I'm using 0, as the caffe ReLU layer ensures coefficients
# are positive and the data is already regularized into the range 0-20.
filled_data = all_data.fillna(0)

# split back into Kaggle train / test sets
filled_train = filled_data.iloc[:len(train)]
filled_test = filled_data.iloc[len(train):]

# split the Kaggle test data into train/test data
X, Xt, y, yt = train_test_split(filled_train.iloc[:, 1:], filled_train.iloc[:, 0], test_size=0.1)
# before submitting results, train on the whole dataset
# X, y = filled_train.iloc[:, 1:], filled_train.iloc[:, 0]
# Xt, yt = filled_train.iloc[:, 1:], filled_train.iloc[:, 0]

dirname = os.path.abspath('./bnp_paribas/data')
train_filename = os.path.join(dirname, 'train.h5')
test_filename = os.path.join(dirname, 'test.h5')
comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}

with h5py.File(train_filename, 'w') as f:
    f.create_dataset('data', data=X, **comp_kwargs)
    f.create_dataset('label', data=y.astype(np.float32), **comp_kwargs)
with h5py.File(test_filename, 'w') as f:
    f.create_dataset('data', data=Xt, **comp_kwargs)
    f.create_dataset('label', data=yt.astype(np.float32), **comp_kwargs)
with open(os.path.join(dirname, 'train.txt'), 'w') as f:
    f.write(train_filename + '\n')
with open(os.path.join(dirname, 'test.txt'), 'w') as f:
    f.write(test_filename + '\n')

# write out the feature engineered test dataset for use in the prediction step
filled_test.to_csv(os.path.join(dirname, 'test_augmented.csv'))
