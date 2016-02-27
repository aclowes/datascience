"""
Run this script _after_ the caffe model has been trained, to generate predictions
 for the Kaggle test data.
"""

import csv
import sys
import numpy as np
import pandas as pd

sys.path.append('../caffe/python')
import caffe

# based on https://kaggle2.blob.core.windows.net/forum-message-attachments/65059/2019/createSubmission2.py

description =      './bnp_paribas/caffe/nonlinear_predict.prototxt'
learned_model =    './bnp_paribas/data/train_iter_3000.caffemodel'
data = pd.read_csv('./bnp_paribas/data/test_augmented.csv', index_col=0)
submission_name =  './bnp_paribas/data/predictions.csv'

headers = ['ID', 'PredictedProb']

# load the pre-trained model
caffe.set_mode_cpu()
net = caffe.Net(description, learned_model, caffe.TEST)

# reshape the data into the format used by Caffe:
# (x, 1, 1, 290) which is (observations, channels, height, width)
prep = data.iloc[:, 1:].as_matrix()[:, np.newaxis, np.newaxis]

# propogate the data through the neural net
results = net.forward_all(data=prep)

# write the submission file by reading the output from the net
with open(submission_name, 'w') as f:
    w = csv.writer(f)
    w.writerow(headers)

    for index, predictions in enumerate(results['prob']):
        w.writerow([data.iloc[index].name, predictions[1]])
