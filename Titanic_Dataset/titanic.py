import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
final_index = test_data.iloc[:,0]

# Extract the train labels
train_labels = train_data.iloc[:,1]

# Convert male, female to binary
train_data.loc[(train_data.iloc[:,4] == 'male'),"Sex"] = 0
train_data.loc[(train_data.iloc[:,4] == 'female'),"Sex"] = 1

test_data.loc[(test_data.iloc[:,3] == 'male'),"Sex"] = 0
test_data.loc[(test_data.iloc[:,3] == 'female'),"Sex"] = 1

# Convert age to bins
train_data.loc[(train_data.iloc[:,5] < 18),"Age"] = 0
train_data.loc[(train_data.iloc[:,5] > 0) & (train_data.iloc[:,5] < 40),"Age"] = 1
train_data.loc[(train_data.iloc[:,5] >= 40),"Age"] = 2
train_data.loc[(train_data.iloc[:,5].isnull()),"Age"] = 1

test_data.loc[(test_data.iloc[:,4] < 18),"Age"] = 0
test_data.loc[(test_data.iloc[:,4] > 0) & (test_data.iloc[:,4] < 40),"Age"] = 1
test_data.loc[(test_data.iloc[:,4] >= 40),"Age"] = 2
test_data.loc[(test_data.iloc[:,4].isnull()),"Age"] = 1

# Null fares
test_data.loc[(test_data.iloc[:,8].isnull()),"Fare"] = 20

# Subset only the important features
train_data = train_data.iloc[:,[2,4,5,6,7,9]]
test_data = test_data.iloc[:, [1,3,4,5,6,8]]

# print(train_data)

clf = RandomForestClassifier(n_estimators=10, min_samples_leaf=5)
clf.fit(train_data, train_labels)

predicted_labels = clf.predict(test_data)

df = pd.concat([final_index, pd.DataFrame(predicted_labels)], axis=1)

df.to_csv("predicted.csv", header=["PassengerId", "Survived"], index=None)
