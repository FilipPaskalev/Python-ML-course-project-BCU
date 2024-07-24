###########################################################################
# Import needed libraries
###########################################################################
import pandas as pd
from sklearn.preprocessing import LabelEncoder

###########################################################################
# Importing data
###########################################################################
source = pd.read_csv('Islander_data.csv')
data = source.copy()

###########################################################################
# Basic data analysis
###########################################################################
print(data.head())
data.info()
data.isnull().sum()
print(data.isnull().sum())

###########################################################################
# Preprocessing the data
###########################################################################
data["name"] = data["first_name"] + " " + data["last_name"]
del data['first_name']e
del data['last_name']

data['Drug'] = data['Drug'].str.replace('A', '0', case = False)
data['Drug'] = data['Drug'].str.replace('T', '1', case = False)
data['Drug'] = data['Drug'].str.replace('S', '2', case = False)
data['Happy_Sad_group'] = data['Happy_Sad_group'].str.replace('H', '0', case = False)
data['Happy_Sad_group'] = data['Happy_Sad_group'].str.replace('S', '1', case = False)

data['Drug'] = data['Drug'].astype(int)
data['Happy_Sad_group'] = data['Happy_Sad_group'].astype(int)

age_group = []
for i in data.itertuples(): 
    if i[1] <= 15:
        age_group.append(0)
    elif i[1] <= 30:
        age_group.append(1)
    elif i[1] <= 50:
        age_group.append(2)
    else:
        age_group.append(3)
data['age_group'] = age_group

label_encoder = LabelEncoder() 
data['name']= label_encoder.fit_transform(data['name'])

###########################################################################
# Optimize dataset 
###########################################################################
data['age_group']= label_encoder.fit_transform(data['age_group'])
data['Drug']= label_encoder.fit_transform(data['Drug'])
data['Happy_Sad_group']= label_encoder.fit_transform(data['Happy_Sad_group'])

del label_encoder
del age_group
del i

data['Dosage'] = data['Dosage'].astype(int)
data['age'] = data['age'].astype(int)
data['age_group'] = data['age_group'].astype(int)

###########################################################################
# Save final data
###########################################################################
data.to_csv('data.csv', index=False)
