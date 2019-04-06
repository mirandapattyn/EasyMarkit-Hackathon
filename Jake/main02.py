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

    jake_folder = "C:\\TemporaryFiles\\"

    X = None
    Y = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None

    model_classification = None

    excluding_columns = ['ReminderId',  # N/A
                         'apt_date',
                         'gender',  # This column is verified to be INEFFECTIVE.
                         'sent_time',  # N/A
                         'apt_type',  # N/A
                         'cli_zip',  # N/A
                         'net_hour',
                         'pat_id',
                         # 'clinic',
                         # 'family_id',
                         ]
# 66.5 67.00 67.10 67.57 67.61
    ############################################################################
    #                              Data Pre-Processing                         #
    ############################################################################

    def __init__(self):
        pass

    def load_trainingset(self, shuffle):
        print("<==== ====", inspect.stack()[0][3], "==== ====>")

        df = pd.read_csv(self.jake_folder + "train.csv")

        print("Prepare the Data Sets")

        # ONE HOT ENCODING.
        df = pd.get_dummies(
            df, columns=[n for n in ['city', 'province', 'type', 'send_time', 'pat_area', 'cli_area']],
            dtype=np.int64
        )

        # For these columns we don't want to use.
        temp = self.excluding_columns
        temp.append('response')
        self.X = df.drop(temp, axis=1)
        temp.remove('response')

        # These fields only exist in the training data, so we have to drop it.
        self.X = self.X.drop(
            ['city_Forest', 'send_time_8:45:00', 'city_Whitecourt', 'city_Salt Spring Island',
             'send_time_11:20:00', 'city_Brantford', 'city_Brockville', 'send_time_14:30:00',
             'city_Laval', 'city_Saint-Constant', 'send_time_18:40:00']
            , axis=1)

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
        print("Start the experiment.")

        # ExtraTree: acc 67.685152
        self.model_classification = sklearn.tree.ExtraTreeClassifier(
            class_weight=None, criterion='gini', max_depth=None,
            max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, random_state=None,
            splitter='random')

        # XGBoost: acc TOO_LONG_TO_EXECUTE
        # self.model_classification = xgb.XGBClassifier()

        self.model_classification = self.model_classification.fit(self.x_train, self.y_train)

        y_pred = self.model_classification.predict(self.x_test)
        self.print_classification_performance_metrics(self.y_test, y_pred)

    def test(self):
        print("Predict the test set.")

        df = pd.read_csv(self.jake_folder + "test.csv")

        print("Prepare the Data Sets")

        # ONE HOT ENCODING.
        df = pd.get_dummies(
            df, columns=[n for n in ['city', 'province', 'type', 'send_time', 'pat_area', 'cli_area']],
            dtype=np.int64
        )
        Y = df.ReminderId

        # For these columns we don't want to use.
        X = df.drop(self.excluding_columns, axis=1)

        x_train, x_test, y_train, y_test = sklearn.model_selection\
            .train_test_split(X, Y, test_size=0.0, shuffle=False)

        y_pred = self.model_classification.predict(x_train)

        # Make the submission file.
        submission = pd.DataFrame(y_pred, columns=['response'])
        submission.to_csv(self.jake_folder + "submission.csv", index=True, index_label='ReminderId')

        # Print out success message.
        print("COMPLETE: submission.csv created!")

################################################################################
#                                 Main                                         #
################################################################################

if __name__ == "__main__":

    dataScienceModeler = DataScienceModeler()
    dataScienceModeler.load_trainingset(False)
    dataScienceModeler.experiment()
    dataScienceModeler.test()

