# EXPERIMENT FOR JUSTIN'S BIG IMPROVEMENT


# Default Libraries
import math
import inspect
import itertools

# Data Science Libraries
import numpy as np
import pandas as pd
import sklearn_pandas

# sklearn Library (https://scikit-learn.org)
import sklearn.model_selection
import sklearn.linear_model
import sklearn.neighbors
import sklearn.metrics
import sklearn.tree
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.neural_network
from sklearn.externals import joblib
import xgboost

class DataScienceModeler:

    r_X = None
    r_Y = None
    r_x_train = None
    r_x_test = None
    r_y_train = None
    r_y_test = None

    c_X = None
    c_Y = None
    c_x_train = None
    c_x_test = None
    c_y_train = None
    c_y_test = None

    TESTSET_X = None

    COMPETITIONSET_X = None

    ############################################################################
    #                              Data Pre-Processing                         #
    ############################################################################

    def __init__(self):
        log_file_name = "log-" + str(pd.Timestamp.now())
        log_file_name = log_file_name.replace(":", "")
        self.log_file_handler = open("logs//" + log_file_name, "w")

    def __del__(self):
        self.log_file_handler.close()

    def load_trainingset(self, shuffle):
        print("<==== ====", inspect.stack()[0][3], "==== ====>")

        df = pd.read_csv("trainingset.csv")

        continuous_features = ["1", "2", "6", "8", "10"]
        categorical_features = ["3", "4", "5", "7", "9", "11", "12",
                                "13", "14", "15", "16", "17", "18"]

        df['Claimed'] = np.where(df['ClaimAmount'] > 0, 1, 0)

        col_list = list(df.columns)
        col_list.remove('rowIndex')
        col_list.remove('Claimed')
        col_list.remove('ClaimAmount')
        col_list.insert(0, 'ClaimAmount')
        col_list.insert(0, 'Claimed')
        col_list.insert(0, 'rowIndex')
        df = df[col_list]

        df = pd.get_dummies(
            df, columns=["feature" + n for n in categorical_features],
            dtype=np.int64
        )

        transform_mapper = sklearn_pandas.DataFrameMapper([
            ('rowIndex', None),
            ('Claimed', None),
            ('ClaimAmount', None),
        ], default=sklearn.preprocessing.StandardScaler())
        standardized = transform_mapper.fit_transform(df.copy())
        df = pd.DataFrame(standardized, columns=df.columns)





        print("0. Prepare the Final Data Sets (Classification)")
        self.c_X = df.drop(['rowIndex', 'Claimed', 'ClaimAmount'], axis=1)
        self.c_Y = df.Claimed
        self.c_x_train, self.c_x_test, self.c_y_train, self.c_y_test = sklearn.model_selection\
            .train_test_split(self.c_X, self.c_Y, test_size=0.30, shuffle=shuffle)

        print("0. Prepare the Final Data Sets (Regression)")
        self.r_X = df.drop(['rowIndex', 'Claimed', 'ClaimAmount'], axis=1)
        self.r_Y = df.ClaimAmount
        self.r_x_train, self.r_x_test, self.r_y_train, self.r_y_test = sklearn.model_selection\
            .train_test_split(self.r_X, self.r_Y, test_size=0.30, shuffle=shuffle)


        print("0. Aggressive Regression")
        df_aggressive_regression = df[: int(0.7 * df.shape[0])]
        df_aggressive_regression = df_aggressive_regression[df_aggressive_regression['ClaimAmount'] > 0]

        print(df_aggressive_regression.shape)
        OUTLIER_CUTOFF = 4647
        df_aggressive_regression = df_aggressive_regression[df_aggressive_regression['ClaimAmount'] < OUTLIER_CUTOFF]

        self.r_x_train_aggressive = df_aggressive_regression.drop(['rowIndex', 'Claimed', 'ClaimAmount'], axis=1)
        self.r_y_train_aggressive = df_aggressive_regression.ClaimAmount

        print(df_aggressive_regression.shape)



    def load_testset(self, shuffle):

        print("<==== ====", inspect.stack()[0][3], "==== ====>")

        df = pd.read_csv("competitionset.csv")

        continuous_features = ["1", "2", "6", "8", "10"]
        categorical_features = ["3", "4", "5", "7", "9", "11", "12",
                                "13", "14", "15", "16", "17", "18"]

        col_list = list(df.columns)
        col_list.remove('rowIndex')
        col_list.insert(0, 'rowIndex')
        df = df[col_list]

        df = pd.get_dummies(
            df, columns=["feature" + n for n in categorical_features],
            dtype=np.int64
        )

        transform_mapper = sklearn_pandas.DataFrameMapper([
            ('rowIndex', None),
        ], default=sklearn.preprocessing.StandardScaler())
        standardized = transform_mapper.fit_transform(df.copy())
        df = pd.DataFrame(standardized, columns=df.columns)

        print("0. Prepare the Final Data Sets (Regression)")
        self.TESTSET_X = df.drop(['rowIndex'], axis=1)

    def load_competitionset(self, shuffle):
        pass

    def select_features_regression(self, num_features):
        print("<==== ====", inspect.stack()[0][3], "==== ====>")

        feature_select_model = sklearn.tree.DecisionTreeRegressor()

        trans = sklearn.feature_selection.RFE(feature_select_model, n_features_to_select=num_features)
        self.r_x_train = trans.fit_transform(self.r_x_train, self.r_y_train)
        self.r_x_test = trans.fit_transform(self.r_x_test, self.r_y_test)

    def select_features_classification(self, num_features):
        print("<==== ====", inspect.stack()[0][3], "==== ====>")

        feature_select_model = sklearn.tree.DecisionTreeClassifier()

        trans = sklearn.feature_selection.RFE(feature_select_model, n_features_to_select=num_features)
        self.c_x_train = trans.fit_transform(self.c_x_train, self.c_y_train)
        self.c_x_test = trans.fit_transform(self.c_x_test, self.c_y_test)

    ############################################################################
    #                                 Utilities                                #
    ############################################################################

    def print_regression_performance_metrics(self, y_test, y_pred):

        label_prediction_difference = np.subtract(y_test, y_pred)
        MAE = np.mean(np.absolute(label_prediction_difference))
        self.log("MAE: " + str(MAE))

        return MAE

    def print_classification_performance_metrics(self, y_test, y_pred):

        confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
        self.log("Confusion Matrix:")
        self.log(str(confusion_matrix))

        # tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test, y_pred).ravel()
        # self.log("TN:", tn, "FP", fp, "FN", fn, "TP", tp)

        f1_score = sklearn.metrics.f1_score(y_test, y_pred)
        self.log("F1 Performance Score: %.6f%%" % (f1_score * 100))

        return f1_score

    ############################################################################
    #                               Model Execution                            #
    ############################################################################

    def experiment00(self):

        self.load_trainingset(False)

        #### #### #### #### Classification Model #### #### #### ####

        model_classification = xgboost.XGBClassifier(learning_rate=0.05, n_estimators=1000,
                                  max_depth=30, min_child_weight=1, gamma=0.1,
                                  subsample=0.8, colsample_bytree=0.8,
                                  objective='binary:logistic', nthread=4,
                                  booster='gbtree', scale_pos_weight=20,
                                  seed=27, reg_lambda=1, reg_alpha=.005)

        model_classification = model_classification.fit(self.c_x_train, self.c_y_train)

        #### #### #### #### Regression Model #### #### #### ####

        # model_regression = xgboost.XGBRegressor(objective='reg:linear', colsample_bytree=0.85,
        #                          eta=0.01, max_depth=9, alpha=10, n_estimators=1,
        #                          booster='gbtree', min_child_weight=0, gamma=0,
        #                          subsample=0.8, reg_alpha=100, max_delta_step=1)
        #
        # model_regression = model_regression.fit(self.r_x_train, self.r_y_train)

        #### #### #### #### Prediction Process #### #### #### ####

        TAU = 0.701
        y_pred_prob = model_classification.predict_proba(self.c_x_test)
        y_pred_prob = pd.DataFrame(y_pred_prob)[1]
        y_pred_classification = \
            y_pred_prob.apply(
                lambda x: 1
                if x > TAU else 0
            ).values

        # y_pred_regression = model_regression.predict(self.r_x_test)
        # y_pred_regression = y_pred_classification * y_pred_regression

        self.print_classification_performance_metrics(self.c_y_test, y_pred_classification)
        # self.print_regression_performance_metrics(self.r_y_test, y_pred_regression)


        # EXP
        y_pred_classification = model_classification.predict(self.c_x_test)
        self.print_classification_performance_metrics(self.c_y_test, y_pred_classification)


    def experiment01(self, PARAMETER_TO_EXPLORE):

        self.load_trainingset(False)

        #### #### #### #### Classification Model #### #### #### ####



        # 0.47
        # model_classification = xgboost.XGBClassifier(learning_rate=0.05, n_estimators=1000,
        #                           max_depth=30, min_child_weight=1, gamma=0.1,
        #                           subsample=0.8, colsample_bytree=0.8, colsample_bylevel=1,
        #                           objective='binary:logistic', nthread=4, n_jobs=2,
        #                           booster='gbtree', scale_pos_weight=20,
        #                           seed=0, reg_lambda=1, reg_alpha=0.1)

        model_classification = xgboost.XGBClassifier(learning_rate=0.05, n_estimators=100,
                                  max_depth=100, min_child_weight=1, gamma=0,
                                  subsample=0.8, colsample_bytree=0.8, colsample_bylevel=1,
                                  objective='binary:logistic', nthread=4, n_jobs=2,
                                  booster='gbtree', scale_pos_weight=20,
                                  seed=0, reg_lambda=1, reg_alpha=0.1,
                                  grow_policy='depthwise', max_leaves=100,
                                  )

        def xgb_f1(y, t):
            t = t.get_label()
            y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y]
            return 'f1', sklearn.metrics.f1_score(t, y_bin)
        model_classification = model_classification.fit(self.c_x_train, self.c_y_train,
                                                        eval_metric=xgb_f1,
                                                        eval_set=[(self.c_x_test, self.c_y_test)],
                                                        verbose=True)


        #### #### #### #### Regression Model #### #### #### ####

        # model_regression = xgboost.XGBRegressor(learning_rate=0.01, n_estimators=100,
        #                          objective='reg:linear', colsample_bytree=0.85,
        #                          max_depth=100, alpha=10,
        #                          booster='gbtree', min_child_weight=0, gamma=0,
        #                          subsample=0.8, reg_alpha=100, max_delta_step=1)
        #
        # model_regression = model_regression.fit(self.r_x_train, self.r_y_train,
        #                                         eval_metric='mae',
        #                                         eval_set=[(self.r_x_test, self.r_y_test)],
        #                                         verbose=True)



        model_regression = sklearn.tree.ExtraTreeRegressor(criterion='mse', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, min_impurity_decrease=0.0,
                           min_impurity_split=None, min_samples_leaf=1,
                           min_samples_split=2,
                           min_weight_fraction_leaf=0.0, random_state=None,
                           splitter='random')


        # model_regression = model_regression.fit(self.r_x_train, self.r_y_train)
        model_regression = model_regression.fit(self.r_x_train_aggressive, self.r_y_train_aggressive)


        #### #### #### #### Prediction Process #### #### #### ####

        TAU_LIST = np.arange(0.001, 0.999, 0.05)

        tau_best = -1
        f1_score_best = -1
        confusion_matrix_bext = ""
        mae_best = -1

        y_pred_prob = model_classification.predict_proba(self.c_x_test)
        y_pred_prob = pd.DataFrame(y_pred_prob)[1]

        for TAU in TAU_LIST:
            self.log("---- ---- ---- ---- TAU:" + str(TAU) + " ---- ---- ---- ----")
            y_pred_classification = \
                y_pred_prob.apply(
                    lambda x: 1
                    if x > TAU else 0
                ).values
            y_pred_regression = model_regression.predict(self.r_x_test)
            y_pred_final = y_pred_classification * y_pred_regression

            f1_current = self.print_classification_performance_metrics(self.c_y_test, y_pred_classification)
            mae_current = self.print_regression_performance_metrics(self.r_y_test, y_pred_final)

            if f1_current > f1_score_best:
                f1_score_best = f1_current
                tau_best = TAU
                confusion_matrix_bext = str(sklearn.metrics.confusion_matrix(self.c_y_test, y_pred_classification))
                mae_best = mae_current


        self.log("<==== ==== ==== ==== REFRENCE - BEST METRICS ==== ==== ==== ====>")
        self.log("Best Tau: " + str(tau_best))
        self.log("Best F1 Score: " + str(f1_score_best))
        self.log(confusion_matrix_bext)
        self.log("Best MAE: " + str(mae_best))

        self.log("<<<< <<<< <<<< <<<< REFRENCE - TRAINING METRICS >>>> >>>> >>>> >>>>")
        y_pred_reference = model_classification.predict(self.c_x_train)
        self.print_classification_performance_metrics(self.c_y_train, y_pred_reference)
        self.log(str(model_classification))

        self.log("<<<< <<<< <<<< <<<< REFRENCE - ALL 0s MAE >>>> >>>> >>>> >>>>")
        y_pred_classification = \
            y_pred_prob.apply(
                lambda x: 1
                if x > tau_best else 0
            ).values

        y_pred_regression = model_regression.predict(self.r_x_test)
        y_pred_regression = pd.DataFrame(y_pred_regression)[0]
        y_pred_regression = \
            y_pred_regression.apply(
                lambda x: 0.000001
                if x > 0 else 0
            ).values

        y_pred_final = y_pred_classification * y_pred_regression
        self.print_regression_performance_metrics(self.r_y_test, y_pred_final)


    def train(self):

        # Load the training set.
        self.load_trainingset(False)

        #### #### #### #### Classification Model #### #### #### ####
        model_classification = xgboost.XGBClassifier(learning_rate=0.19, n_estimators=800,
                                  max_depth=20, min_child_weight=1, gamma=0.1,
                                  subsample=0.8, colsample_bytree=0.8,
                                  objective='binary:logistic', nthread=4,
                                  booster='gbtree', scale_pos_weight=20,
                                  seed=27, reg_alpha=.005)
        model_classification = model_classification.fit(self.c_x_train, self.c_y_train)

        #### #### #### #### Regression Model #### #### #### ####
        model_regression = xgboost.XGBRegressor(objective='reg:linear', colsample_bytree=0.85,
                                 eta=0.01, max_depth=9, alpha=10, n_estimators=1,
                                 booster='gbtree', min_child_weight=0, gamma=0,
                                 subsample=0.8, reg_alpha=100, max_delta_step=1)
        model_regression = model_regression.fit(self.r_x_train, self.r_y_train)

        #### #### #### #### Save Models #### #### #### ####
        joblib.dump(model_classification, 'model_classification')
        joblib.dump(model_regression, 'model_regression')

    def assess(self):

        # Load the test set.
        self.load_testset(False)

        #### #### #### #### Prediction Process #### #### #### ####
        model_classification = joblib.load('model_classification')
        model_regression = joblib.load('model_regression')

        #### #### #### #### Prediction Process #### #### #### ####
        TAU = 0.701
        y_pred_prob = model_classification.predict_proba(self.TESTSET_X)
        y_pred_prob = pd.DataFrame(y_pred_prob)[1]
        y_pred_classification = \
            y_pred_prob.apply(
                lambda x: 1
                if x > TAU else 0
            ).values
        y_pred_regression = model_regression.predict(self.TESTSET_X)
        y_pred_final = y_pred_classification * y_pred_regression

        #### #### #### #### EXPERIMENT!!!!!!! #### #### #### ####
        y_pred_final = pd.DataFrame(y_pred_final)[0]
        y_pred_final = \
            y_pred_final.apply(
                lambda x: 0.000001
                if x != 0.0 else 0
            ).values

        # Make the submission file.
        submission = pd.DataFrame(y_pred_final, columns=['ClaimAmount'])
        submission.to_csv("submission.csv", index=True, index_label='rowIndex')

        # Print out success message.
        print("COMPLETE: submission.csv created!")


    def log(self, message):
        self.log_file_handler.write(message + "\n")
        print(message)


################################################################################
#                                 Main                                         #
################################################################################

if __name__ == "__main__":

    DataScienceModeler().experiment01(0.1)



    # DataScienceModeler().train()
    # DataScienceModeler().assess()


