


import pandas as pd
import numpy as np
from sklearn import preprocessing
raw_race = pd.read_csv("rpscrape1/data/flat/gb/2013-2019.csv")

#Create individual race ID's
races = raw_race["Date"] + raw_race["Course"] + raw_race["Off"]
raw_race.insert(0,column = "raceID", value = races)
le = preprocessing.LabelEncoder()
raw_race["raceID"] = le.fit_transform(raw_race["raceID"])
#Create unique row IDs
raw_race.insert(1,"UID",raw_race["raceID"].astype(str) + raw_race["Horse"])
raw_race["Dist_F"] = raw_race["Dist_F"].apply(lambda x : x[:-1]).astype("double")
raw_race = raw_race.sort_values(by=["Date","Off"]).reset_index(drop = True)


#Get track direction
r_handed = [
    "Ascot", "Beverley","Carlisle","Goodwood","Hamilton","Kempton-AW",
    "Leicester", "Musselburgh","Newmarket","Newmarket-July","Ripon",
    "Sailsbury","Sandown","Windsor"
    ]
raw_race["Track_direction"] = np.where(raw_race["Course"].isin(r_handed), "R","L")

gping = [
    "Ascot","Ayr","Bath","Chelmsford-AW","Doncaster","Ffos-Las","Haydock","Leicester","Newbury",
    "Newcastle-AW","Newcastle","Newmarket-July","Newmarket","Nottingham","Redcar","Salisbury","Wetherby","Windsor",
    "Yarmouth","York","Lingfield"

]
stiff = [
    "Beverley", "Brighton","Carlisle","Chepstow","Epsom","Goodwood","Hamilton","Musselburgh","Pontefract",
    "Sandown","Southwell-AW","Thirsk","Warwick"
]
tight = [
    "Catterick","Chester","Lingfield-AW","Ripon","Wolverhampton-AW","Kempton-AW"

]

raw_race.loc[raw_race["Course"].isin(gping),"Track_speed"] = "Galloping"
raw_race.loc[raw_race["Course"].isin(stiff),"Track_speed"] = "Stiff"
raw_race.loc[raw_race["Course"].isin(tight),"Track_speed"] = "Tight"



#Some NA filling
raw_race["Prize"].fillna(0, inplace = True)
raw_race["Sex_Rest"].fillna("None", inplace = True)
raw_race["HG"].fillna("None", inplace = True)
raw_race["Rating_Band"].fillna("None_handicap", inplace = True)
raw_race["Pattern"].fillna("None", inplace = True)
raw_race["Going"].fillna("Standard", inplace = True)
raw_race["OR"].fillna(raw_race.groupby("Horse")["RPR"].shift(1), inplace = True)
raw_race["RPR"].fillna(raw_race["OR"], inplace = True)


#Recode class and pattern
raw_race["Class"] = raw_race["Class"].map({"Class 1":1,"Class 2":2,"Class 3":3,"Class 4":4,"Class 5":5,"Class 6":6,"Class 7":7 })
raw_race["Pattern"] = raw_race["Pattern"].map({"Group 1":1,"Group 2":2,"Group 3":3,"Listed":4,"None":5})

#Drop rows with missing values in these columns
raw_race.dropna(subset = ["Draw","Horse","SP","Dec","Damsire","Num"],inplace = True)

#Impute missing OR or RPR vales with mean of horses in same class and same age
raw_race["OR"].fillna(raw_race.groupby(["Age","Class"])["OR"].transform('mean'), inplace = True)
raw_race["RPR"].fillna(raw_race.groupby(["Age","Class"])["RPR"].transform('mean'), inplace = True)
race_data = raw_race.drop(["Dist","Dist_Y","Num","SP","Wgt"],axis = 1)

race_data.to_csv("/Users/leomccarthy/Documents/Projects/Horse_racing/rpscrape1/data/flat/gb/2013_2019_clean.csv", index = False)
