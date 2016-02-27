# Kaggle BNP Paribas Challenge

https://www.kaggle.com/c/bnp-paribas-cardif-claims-management

## Caffe

[Caffe][1] is a neural network deep learning library created
at Berkeley. It is popular for computer vision challenges and
research, including ImageNet.

It can also be used for classification problem such as this!
Here's how to run it:

- `bnp_paribas/caffe_prepare.py` splits categorical columns in
 the source data into multiple binary columns, and saves the data
 into the HDF5 format used by Caffe.
- `../caffe/build/tools/caffe train -solver bnp_paribas/caffe/nonlinear_solver.prototxt`
 runs caffe to train the model.
- `bnp_paribas/caffe_predict.py` uses the model to predict
 classification probabilities for the test data and
 outputs them to a CSV.

[1]: http://caffe.berkeleyvision.org/

## Sklearn

Scikit Learn comes with a number of [classification models][2]:
- linear, including SGDClassifier and SVC
- decision trees, including DecisionTreeClassifier and
 RandomForestClassifier

A simple SGDClassifier is implemented in
`bnp_paribas/sklearn_predict.py`

[2]: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html