import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import climate
import theanets
import pylab as pb

def create_datasets():
    fp = "yacht_hydrodynamics.csv"
    df = pd.read_csv(fp)
    col_names = ["LP", "PC", "LDR", "BDR", "LBR", "FN", "RR"]
    df.columns = col_names
    X = df.ix[:, 0:6]
    Y = df.ix[:, 6]
    X = X.as_matrix()
    Y = np.reshape(Y, (Y.shape[0],1))
    #X_scaled = preprocessing.scale(X)
    dp = np.hstack((X,Y))
    train, test = train_test_split(dp, test_size = 0.3)
    
    return train, test

def do_regression():
    climate.enable_default_logging()

    train, test = create_datasets()
    x_train = train[:,0:6]
    y_train = train[:,6]
    y_train = np.reshape(y_train, (y_train.shape[0],1))
    y_test = test[:,6]
    y_test = np.reshape(y_test, (y_test.shape[0],1))
    exp = theanets.Experiment(
    theanets.Regressor,
    layers=(6, 6, 1))
    exp.train([x_train, y_train])

    #do the testing
    x_test = test[:,0:6]
    y_test = test[:, 6]

    yp = exp.network.predict(x_test)
    
    xi = [ (i+1) for i in range(x_test.shape[0])]
    
    
    pb.scatter(xi, y_test, color = "red")
    pb.scatter(xi, yp, color = "blue")

    pb.show()
    

    return

    
    
