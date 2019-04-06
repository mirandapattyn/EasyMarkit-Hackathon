# Default Libraries
import math
import inspect
import itertools

# Data Science Libraries
import numpy as np
import pandas as pd
import sklearn_pandas
import xgboost as xgb

# sklearn Library (https://scikit-learn.org)
import sklearn.model_selection
import sklearn.linear_model
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.metrics
import sklearn.tree
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.neural_network
from sklearn.externals import joblib

class DataScienceModeler:

    X = None
    Y = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None

    ############################################################################
    #                              Data Pre-Processing                         #
    ############################################################################

    def __init__(self):
        pass

    def load_trainingset(self, shuffle):
        print("<==== ====", inspect.stack()[0][3], "==== ====>")

        df = pd.read_csv("C:\\TemporaryFiles\\train.csv")

        print("0")

        df = pd.get_dummies(
            df, columns=[n for n in ['city', 'province', 'type', 'gender', 'send_time', 'pat_area', 'cli_area']],
            dtype=np.int64
        )

        print("1")
        X_Training = df.drop(['ReminderId', 'response',  # N/A
                          'apt_date',
                          'sent_time',  # N/A
                          'apt_type',  # N/A
                          'net_hour', 'cli_zip'  # N/A
                          ], axis=1)


        print("2")
        #######################################################################################

        df = pd.read_csv("C:\\TemporaryFiles\\test.csv")

        print("3")
        df = pd.get_dummies(
            df, columns=[n for n in ['city', 'province', 'type', 'gender', 'send_time', 'pat_area', 'cli_area']],
            dtype=np.int64
        )

        print("4")
        X_Testing = df.drop(['ReminderId',  # N/A
                          'apt_date',
                          'sent_time',  # N/A
                          'apt_type',  # N/A
                          'net_hour', 'cli_zip'  # N/A
                          ], axis=1)

        print("5")
        print(X_Training.columns)
        print(X_Testing.columns)

        X_Training_Set = set()
        for col in X_Training.columns:
            X_Training_Set.add(col)

        X_Testing_Set = set()
        for col in X_Testing.columns:
            X_Testing_Set.add(col)


        print("Only in Training:")
        print(X_Training_Set.difference(X_Testing_Set))

        print("Only in Testing:")
        print(X_Testing_Set.difference(X_Training_Set))





################################################################################
#                                 Main                                         #
################################################################################

if __name__ == "__main__":

    dataScienceModeler = DataScienceModeler()
    dataScienceModeler.load_trainingset(False)






