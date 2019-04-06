import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def kfoldCV(x, y, k):
    fold_errors = []
    subset_len = int(len(x) / 10)

    for i in range(10):
        val_x = x[i * subset_len: i * subset_len + subset_len]
        val_y = y[i * subset_len: i * subset_len + subset_len]

        train_x = np.delete(x, slice(i * subset_len, i * subset_len + subset_len), axis=0)
        train_y = np.delete(y, slice(i * subset_len, i * subset_len + subset_len), axis=0)

        individual_errors = []

        knc = KNeighborsClassifier(n_neighbors=int(k))

        knc.fit(train_x, train_y)

        for j in range(len(val_x)):
            output = knc.predict([val_x[j]])
            individual_errors.append(output == val_y[j])

        fold_error = individual_errors.count(False)/len(individual_errors)
        print(fold_error)
        fold_errors.append(fold_error)

    return sum(fold_errors)/len(fold_errors)


# PART 3
data = pd.read_csv("data/train.csv")

Y = data.response

data.drop(['ReminderId', 'response', 'apt_type', 'apt_date', 'sent_time', 'clinic', 'city', 'province',
             'cli_zip', 'cli_area', 'cli_size', 'pat_id', 'family_id', 'fam', 'pat_area'],
            axis=1, inplace=True)

df_new = pd.get_dummies(data) # send_time, type, net_hour, gender, age, dist

X = df_new

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, shuffle=False, random_state=0)

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

k_values = np.linspace(1, 3, num=2)
print(k_values)
errors = []

for i in k_values:
    error = kfoldCV(train_x, train_y, i)
    errors.append(error)

print(errors)

best_k = k_values[errors.index(min(errors))]

print("the best k is " + str(best_k))

best_k_errors = []

knc = KNeighborsClassifier(n_neighbors=int(best_k))

knc.fit(train_x, train_y)

for i in range(len(test_x)):
    best_output = knc.predict([test_x[i]])
    best_k_errors.append(best_output == test_y[i])

print("the error rate on the test set with the best k is " + str(best_k_errors.count(False)/len(best_k_errors)))
