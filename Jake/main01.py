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

        print("0. Prepare the Data Sets")

        df = pd.get_dummies(
            df, columns=[n for n in ['city', 'province', 'type', 'gender', 'send_time', 'pat_area', 'cli_area']],
            dtype=np.int64
        )

        self.X = df.drop(['ReminderId', 'response',  # N/A
                          'apt_date',
                          'sent_time',  # N/A
                          'apt_type',  # N/A
                          'net_hour', 'clinic', 'cli_zip', 'pat_id', 'family_id'  # N/A
                          ], axis=1)
        self.Y = df.response

        self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection\
            .train_test_split(self.X, self.Y, test_size=0.30, shuffle=shuffle)



    ############################################################################
    #                                 Utilities                                #
    ############################################################################

    def print_classification_performance_metrics(self, y_test, y_pred):

        confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(confusion_matrix)

        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test, y_pred).ravel()
        print("TN:", tn, "FP", fp, "FN", fn, "TP", tp)

        f1_score = sklearn.metrics.f1_score(y_test, y_pred)
        print("F1 Performance Score: %.6f%%" % (f1_score * 100))

        acc = sklearn.metrics.accuracy_score(y_test, y_pred)
        print("Accuracy Performance Score: %.6f%%" % (acc * 100))




    ############################################################################
    #                               Model Execution                            #
    ############################################################################

    def experiment(self):

        model_classification = sklearn.tree.ExtraTreeClassifier(
            class_weight=None, criterion='gini', max_depth=None,
            max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, random_state=None,
            splitter='random')

        # model_classification = xgb.XGBClassifier().fit(self.x_train, self.y_train)
        model_classification = model_classification.fit(self.x_train, self.y_train)

        y_pred = model_classification.predict(self.x_test)
        self.print_classification_performance_metrics(self.y_test, y_pred)


################################################################################
#                                 Main                                         #
################################################################################

if __name__ == "__main__":

    dataScienceModeler = DataScienceModeler()
    dataScienceModeler.load_trainingset(False)
    dataScienceModeler.experiment()





