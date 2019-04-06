import pandas as pd
import xgboost as xgb
import winsound
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer


def main():

    ### Train set ###

    df = pd.read_csv("train.csv")

    Y = df.response

    df.drop(['ReminderId', 'response', 'apt_type', 'apt_date', 'sent_time', 'clinic', 'city', 'province',
             'cli_zip', 'cli_area', 'cli_size', 'pat_id', 'family_id', 'fam', 'pat_area'],
            axis=1, inplace=True)

    df_new = pd.get_dummies(df) # send_time, type, net_hour, gender, age, dist

    X = df_new

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False, random_state=0)

    model = xgb.XGBClassifier()

    model.fit(X_train, Y_train)

    result = model.predict(X_test)

    print("Train set Accuracy:", accuracy_score(Y_test, result))

    # ### Test set ###
    #
    # df_test = pd.read_csv("test.csv")
    #
    # df_test.drop(['ReminderId', 'apt_type', 'apt_date', 'sent_time', 'clinic', 'city', 'province',
    #          'cli_zip', 'cli_area', 'cli_size', 'pat_id', 'family_id', 'fam', 'pat_area'],
    #         axis=1, inplace=True)
    #
    # df_test_new = pd.get_dummies(df_test)  # send_time, type, net_hour, gender, age, dist
    #
    # X = df_test_new.values
    #
    # result_test = model.predict(X)
    #
    # print("Test set Accuracy:", accuracy_score(Y_test, result_test))


if __name__ == "__main__":
    main()

