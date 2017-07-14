import pandas as pd
from sklearn.datasets import load_boston

# boston is a dictionary

boston = load_boston()

# Store the data and target in respective variables

data = boston.data
target = boston.target

# Convert to pandas dataframe and assign column names to dataframe

(data, target) = (pd.DataFrame(data), pd.DataFrame(target))
data.columns = boston.feature_names

# Print the shape of the data

print('The shape of the data is : ' + str(data.shape))

# Find dependence of data on attribute "CHAS", see description
# of data to see what it represents
# Calculate the average price for houses "not near" and "near" the river Charles("CHAS")

chas = [0, 0]
chas[0] = target[np.array(data['CHAS'] == 0)][0].mean()
chas[1] = target[np.array(data['CHAS'] == 1)][0].mean()
