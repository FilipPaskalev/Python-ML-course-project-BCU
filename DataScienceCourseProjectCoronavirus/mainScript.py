#Description row by row

#Load pandas library for reading/writing csv files
import pandas as panda

#create data file with all data
#dataFrame -> df
df = panda.read_csv("2019_nC0v_20200121_20200126_cleaned.csv")

#check is it all columns have names
#print(df.columns)

#Create mirror file for operating over it. File is in project dir
#modifyDataFile -> mdf
df.to_csv("modifyDataFile.csv")

#create file
mdf = panda.read_csv("modifyDataFile.csv")

#remove empty columns because they doesn't have a name 
#and the value is unnecessary,it's just indexing
mdf.drop(["Unnamed: 0","Unnamed: 0.1"], axis = 1, inplace = True) 

#check is it all columns have names
print(mdf.columns)

mdf = mdf.rename(columns={"Province/State":"Province"})