import pandas as pd
import xgboost as xgb
import winsound
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer


def xgboost_classifier(X_train, X_test, Y_train):

    print("Entered classifier method.")

    model = xgb.XGBClassifier()

    start = timer()
    print("Start fitting")
    model.fit(X_train, Y_train)
    end = timer()
    print("Took", end - start, "seconds")

    start = timer()
    print("Start predicting")
    preds = model.predict(X_test)
    end = timer()
    print("Took", end - start, "seconds")

    return preds


def main():

    df = pd.read_csv("train.csv")

    Y = df.response

    df.drop(['ReminderId', 'response', 'apt_type', 'apt_date', 'sent_time', 'clinic', 'city', 'province',
             'cli_zip', 'cli_area', 'cli_size', 'pat_id', 'family_id', 'fam', 'pat_area'],
            axis=1, inplace=True)

    df_new = pd.get_dummies(df) # send_time, type, net_hour, gender, age, dist

    X = df_new

    start = timer()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False, random_state=0)

    result = xgboost_classifier(X_train, X_test, Y_train)

    print("Train set Accuracy:", accuracy_score(Y_test, result))
    end = timer()
    print("Took", end - start, "seconds")

    winsound.Beep(250, 1000)


if __name__ == "__main__":
    main()

