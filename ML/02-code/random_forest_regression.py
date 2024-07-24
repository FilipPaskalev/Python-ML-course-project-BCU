###########################################################################
# Imports
###########################################################################
import numpy as np
import pandas as pd
import sklearn as sklearn
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle 
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

###########################################################################
# Data
###########################################################################
data = pd.read_csv('data.csv')

data = shuffle(data)

data = data[[
"age",
"Happy_Sad_group",
"Dosage",
"Drug",
"Mem_Score_Before",
"Mem_Score_After",

"name",
"age_group"
]]

predict = "age_group"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

###########################################################################
# Separate data
###########################################################################
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1, random_state = 0)

###########################################################################
# Train
###########################################################################
best = 0
for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    forest = RandomForestRegressor()

    forest.fit(x_train, y_train)
    acc = forest.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("random_forest_regression.pickle", "wb") as f:
            pickle.dump(forest, f)

###########################################################################
# Load model
###########################################################################
pickle_in = open("random_forest_regression.pickle", "rb")
forest = pickle.load(pickle_in)

###########################################################################
# Predict
###########################################################################
predicted = forest.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])
    
###########################################################################
# Observationo, charts, plotting model
###########################################################################
plt.style.use('fivethirtyeight')

# plt.tight_layout()
# sns.distplot(data['age_group'])

# plot = "Dosage"
# plt.scatter(data[plot], data["Mem_Score_After"], )
# plt.legend(loc=4)
# plt.title("Dosage VS Mem_Score_After")
# plt.xlabel(plot)
# plt.ylabel("Mem_Score_After")
# plt.show()

# fig=plt.figure(figsize=(18,18))
# sns.heatmap(data.corr(), annot= True, cmap='Blues')

# fig = plt.figure(figsize=(12, 5))
# ax = fig.add_axes([0,0,1,1])
# sns.countplot(x='age', data=data, ax=ax)
# plt.title('Number of Observations by Age')
# plt.tight_layout()
# plt.show()


# sns.distplot(data.Mem_Score_Before, hist = False, kde = True, norm_hist = True, kde_kws={'linestyle':':'})
# sns.distplot(data.Mem_Score_After, hist = False, kde = True, norm_hist = True, kde_kws={'linestyle':'--'});
# plt.title('Memory Score Before and After the treatment \nBOTH DRUGS')
# plt.text(100, 0.015, '--- AFTER\n... BEFORE');
