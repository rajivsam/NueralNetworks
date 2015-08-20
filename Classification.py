import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import climate
import theanets
import pylab as pb
from sklearn.metrics import confusion_matrix

recode_map = {1: 0, 2: 1, 3: 2, 5: 3, 6: 4, 7: 5}

def recode(x):
    return recode_map[x]



def do_classification():
    
    climate.enable_default_logging()
    df = pd.read_csv("glass.csv")
    #recode the class variable to go from 0 through 5
    df["GT"] = map(recode, df["GT"])
    train, valid = train_test_split(df, test_size = 0.3)
    train_X = train.ix[:, 0:9].as_matrix()
    train_Y = train.ix[:, 9].as_matrix()
    valid_X = valid.ix[:, 0:9].as_matrix()
    valid_Y = valid.ix[:,9].as_matrix()
    train_X = train_X.astype('f')
    train_Y = train_Y.astype('i')
    valid_X = valid_X.astype('f')
    valid_Y = valid_Y.astype('i')
    t0 = (train_X, train_Y)
    t1 = (valid_X, valid_Y)

    exp = theanets.Experiment(theanets.Classifier, layers = (9,18,18,6))
    exp.train(t0, t1, algorithm='sgd',\
              learning_rate=1e-4, momentum=0.1,\
              hidden_l1=0.001, weight_l2=0.001)

    cm = confusion_matrix(valid_Y, exp.network.predict(valid_X))

    return cm
    
