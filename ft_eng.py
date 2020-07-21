import pandas as pd
import numpy as np

race_data = pd.read_csv("rpscrape1/data/flat/gb/2013_2019_clean.csv")
#Remove non finishers
didnt_finish = ["PU","SU","F","LFT","RR","RO","REF","BD","0","UR","DSQ"]
race_data = race_data.loc[~race_data.Pos.isin(didnt_finish)]
race_data["Pos"] = race_data["Pos"].astype(int)
race_data["Won"] = np.where(race_data["Pos"] == 1,1,0)
#Change data data type
race_data["Date"] = race_data["Date"].astype('datetime64')
#Sort frame by date and time for easy manipulation
race_data = race_data.sort_values(by=["Date","Off"]).reset_index(drop = True)




####### Track features ########

#Turn finishing time to seconds
def time2int(t):
    return int(t.split(':')[0]) * 60 + int(t.split('.')[0].split(":")[1]) + int(t.split('.')[1])/100
race_data["Time_secs"] = race_data["Time"].map(time2int)


#Get track/dist/class record
race_data["Course_dist_record"] = race_data.groupby(["Course","Dist_F","Class"])["Time_secs"].apply(lambda x: x.expanding().min().shift())
race_data = race_data.dropna(subset = ["Course_dist_record"]).reset_index(drop = True)

#Wins from draw at course at dist
race_data["Draw_wins"] = race_data.groupby(["Course","Draw","Dist_F"])["Won"].apply(lambda x:x.cumsum().shift().fillna(0))
race_data["Draw_runs"] = race_data.groupby(["Course","Draw","Dist_F"]).cumcount()
race_data["Draw_AE"] = (race_data["Draw_wins"]/(race_data.groupby(["Course","Draw","Dist_F"])["Dec"].apply(lambda x: (1/x).cumsum().shift()) ) ).fillna(0)
race_data["Draw_SR"] = (race_data["Draw_wins"]/race_data["Draw_runs"]).fillna(0)



####### Horse features #######
#Get course-distance-class record for each race at time of race and the time#
race_data["Time_off_cd_record"] = race_data["Time_secs"] - race_data["Course_dist_record"]
#For each horse in each race get the time behind the winner
race_data["Time_off_winner"] = race_data.groupby(["raceID"])["Time_secs"].apply(lambda x: x - x.min())

#Create metrics for quality of horse run
race_data["Time__perf_rating"] = ((race_data["Time_off_winner"]+1)**0.5) * race_data["Class"] * (race_data["Pattern"]**0.5) * (2**race_data["Draw_SR"])
race_data["Btn_perf_rating"] = (((race_data["Ovr_Btn"]+1)**0.5) * race_data["Class"] * (race_data["Pattern"]**0.5) * (2**race_data["Draw_SR"])).round(4)


race_data["Highest_prev_or"] = race_data.groupby("Horse")["OR"].apply(lambda x: x.expanding().max().shift().fillna(0))
race_data["Highest_prev_rpr"] = race_data.groupby("Horse")["RPR"].apply(lambda x: x.expanding().max().shift().fillna(0))
race_data["First_race"] = np.where(race_data["UID"].isin(race_data.groupby("Horse").nth(0).reset_index()["UID"]),1,0)



# Total races combined stats (at time of race)

def get_groups(by):
    return [[by],[by,"Class"], [by,"Going"],[by,"Dist_F"],[by,"Class","Pattern"],[by,"Course"]]


def get_stat(stat,groupby,count):
    if count == True:
        return race_data.groupby(groupby).cumcount()
    else:
        return race_data.groupby(groupby)[stat].apply(lambda x:x.cumsum().shift().fillna(0))

for group in get_groups("Horse"):
    race_data['_'.join(group)+"_count"] = get_stat("none",group,count = True)
    race_data['_'.join(group)+"_" + "Won"] = get_stat("Won",group,count = False)
    race_data['_'.join(group)+"_" + "Prize"] = get_stat("Prize",group,count = False)

for group in get_groups("Horse"):
    race_data['_'.join(group)+"_" + "WinSR"] = race_data['_'.join(group)+"_" + "Won"]/race_data['_'.join(group)+"_" + "count"]
    race_data['_'.join(group)+"_" + "PPR"] = race_data['_'.join(group)+"_" + "Prize"]/race_data['_'.join(group)+"_" + "count"]



#Extract features from text

race_data["Held_up"] = np.where(race_data["Comment"].str.contains("held up", case = False),1, 0)
race_data["Slow_into_stride"] = np.where(race_data["Comment"].str.contains("slowly into stride|started slowly", case = False),1, 0)
race_data["Keen"] = np.where(race_data["Comment"].str.contains("keen|keenly", case = False),1, 0)
race_data["Hampered"] = np.where(race_data["Comment"].str.contains("hampered", case = False),1, 0)
race_data["Stayed_on"] = np.where(race_data["Comment"].str.contains("stayed on|kept on", case = False),1, 0)
race_data["Missed_break"] = np.where(race_data["Comment"].str.contains("missed break|dwelt", case = False),1, 0)
race_data["Weakened"] = np.where(race_data["Comment"].str.contains("weakened|no impression", case = False),1, 0)
race_data["Raced_towards_front"] = np.where((race_data["Comment"].str.contains("chased leaders", case = False))| ((race_data["Comment"].str.find("Led") <  6) & (race_data["Comment"].str.find("Led") > -1)) ,1, 0)

#Last 4 race stats
get_last_4 = ["Won","Prize","Ovr_Btn","Time_off_winner",
              "Time_off_cd_record","Time__perf_rating","Btn_perf_rating",
              "Held_up","Slow_into_stride","Keen","Hampered","Stayed_on",
              "Missed_break","Weakened","Raced_towards_front"]

race_data[[i + "_last4" for i in get_last_4]] = race_data.groupby("Horse")[get_last_4].apply(lambda x : x.rolling(4,min_periods = 1).sum().shift())

# within race ranking features
    # race_data["Total_wins_horse_rank"]  = race_data.groupby("raceID")["Total_wins_horse"].rank(axis = 0, ascending = False,method = "dense")
    # race_data["Total_sr_horse_rank"]  = race_data.groupby("raceID")["Total_sr_horse"].rank(axis = 0, ascending = False,method = "dense")
    # race_data["Total_prize_horse_rank"]  = race_data.groupby("raceID")["Total_prize_horse"].rank(axis = 0, ascending = False,method = "dense")
    #  race_data["Prize_per_race_horse_rank"]  = race_data.groupby("raceID")["Prize_per_race_horse"].rank(axis = 0, ascending = False,method = "dense")
    # race_data["OR_horse_rank"]  = race_data.groupby("raceID")["OR"].rank(axis = 0, ascending = False,method = "dense")
    # race_data["RPR_horse_rank"]  = race_data.groupby("raceID")["RPR"].rank(axis = 0, ascending = False,method = "dense")
    # race_data["Lbs_horse_rank"]  = race_data.groupby("raceID")["Lbs"].rank(axis = 0, ascending = False,method = "dense")



## last race features
last_race_features = [
        "Date","Course","Class","Pattern",
        "Dist_F","Rating_Band","Prize","Track_speed",
        "Horse","Time_secs","Pos","Lbs","Ovr_Btn",
        "Comment","Trainer","Dec","HG","Owner","Won","Time_off_cd_record","Time_off_winner",
        "Time__perf_rating","Btn_perf_rating",
        "Held_up","Slow_into_stride","Keen","Hampered","Stayed_on",
        "Missed_break","Weakened","Raced_towards_front","RPR"]

#Seperate into sperate DF for each horse and shift to creat lag variables
lastDF = race_data.groupby("Horse")[last_race_features].shift(1)
lastDF.columns += "_last_race"
#Merg with origional dataframe
race_data = pd.merge(race_data,lastDF, how = "left",left_index=True, right_index=True)

#Change of distance features
race_data["up_in_trip"] = np.where(race_data["Dist_F"] > (race_data["Dist_F_last_race"] ) ,1,0)
race_data["down_in_trip"] = np.where(race_data["Dist_F"] < (race_data["Dist_F_last_race"] ) ,1,0)
race_data["dist_change"] = (race_data["Dist_F"] -race_data["Dist_F_last_race"]).fillna(0)

#Amount of time off features
race_data["days_since_last"] = ((race_data["Date"] - race_data["Date_last_race"]).dt.days).fillna(0)

#Hedgear change
race_data["HG_last_race"].fillna("None", inplace = True)
race_data["HGchange"] = np.where(race_data["HG"] != race_data["HG_last_race"],1,0)

#Change of class features
race_data["ChangeClass"] = (race_data["Class"] - race_data["Class_last_race"]).fillna(0)

#Change of owner or Trainer
race_data["new_trainer"] = np.where((race_data["Trainer"]== race_data["Trainer_last_race"])|(race_data["Trainer_last_race"].isna()), 0, 1)
race_data["new_owner"] = np.where((race_data["Owner"]== race_data["Owner_last_race"])|(race_data["Owner_last_race"].isna()), 0, 1)

### Sire, Dam, Damsire Stats ###
def get_groups_ped(by):
    return [[by],[by,"Class"], [by,"Going"],[by,"Dist_F"],[by,"Age"],[by,"Course"],[by,"Track_speed"]]

for i in ["Dam","Sire"]:
    for group in get_groups_ped(i):
        race_data['_'.join(group)+"_count"] = get_stat("none",group,count = True)
        race_data['_'.join(group)+"_" + "Won"] = get_stat("Won",group,count = False)
        race_data['_'.join(group)+"_" + "Prize"] = get_stat("Prize",group,count = False)

    for group in get_groups_ped(i):
        race_data['_'.join(group)+"_" + "WinSR"] = race_data['_'.join(group)+"_" + "Won"]/race_data['_'.join(group)+"_" + "count"]
        race_data['_'.join(group)+"_" + "PPR"] = race_data['_'.join(group)+"_" + "Prize"]/race_data['_'.join(group)+"_" + "count"]


######## Trainer Jockey Owner stats ########

def get_groups_t(by):
    return [[by],[by,"Class"],[by,"Course"]]

for i in ["Trainer","Jockey","Owner"]:
    for group in get_groups_t(i):
        race_data['_'.join(group)+"_count"] = get_stat("none",group,count = True)
        race_data['_'.join(group)+"_" + "Won"] = get_stat("Won",group,count = False)
        race_data['_'.join(group)+"_" + "Prize"] = get_stat("Prize",group,count = False)
    for group in get_groups_t("Trainer"):
        race_data['_'.join(group)+"_" + "WinSR"] = race_data['_'.join(group)+"_" + "Won"]/race_data['_'.join(group)+"_" + "count"]
        race_data['_'.join(group)+"_" + "PPR"] = race_data['_'.join(group)+"_" + "Prize"]/race_data['_'.join(group)+"_" + "count"]

for i in ["Trainer","Jockey","Owner"]:
    race_data[i +'_AE'] = (race_data[i+'_Won']/(race_data.groupby(i)["Dec"].apply(lambda x: (1/x).cumsum().shift()) ) ).fillna(0)


#Last 28 and 14 day stats for trainer, Jockey

def get_groups_timed(by):
    return [[by],[by,"Class"]]

# def get_stat_timed(df,stat,groupby,window,count):
#     if count == True:
#         race_data.set_index("Date").groupby(group, sort  = False)["raceID"].apply(lambda x: x.resample("1d").count().rolling('28d').sum().shift().fillna(0))
#     else:
#         race_data.set_index("Date").groupby(groupby, sort  = False)["raceID"].apply(lambda x: x.resample("1d").count().rolling('28d').sum().shift().fillna(0))

for i in ["Trainer","Jockey"]:
    for group in get_groups_timed(i):
        race_data = race_data.merge(race_data.set_index("Date").groupby(group, sort  = False)["raceID"].apply(lambda x: x.resample("1d").count().rolling('28d').sum().shift().fillna(0)),how = "left", on = [j for i in [group,["Date"]] for j in i], suffixes = ("","_"+'_'.join(group)+"_28d_count"))
        for s in ["Won","Prize","Dec"]:
            race_data = race_data.merge(race_data.set_index("Date").groupby(group, sort  = False)[s].apply(lambda x: x.resample("1d").sum().rolling('28d').sum().shift().fillna(0)),how = "left", on = [j for i in [group,["Date"]] for j in i], suffixes = ("","_"+'_'.join(group)+"_28d_sum"))
        for m in ["Prize","Time__perf_rating","Ovr_Btn"]:
            race_data = race_data.merge(race_data.set_index("Date").groupby(group, sort  = False)[m].apply(lambda x: x.resample("1d").mean().rolling('28d').mean().shift().fillna(0)),how = "left", on = [j for i in [group,["Date"]] for j in i], suffixes = ("","_"+'_'.join(group)+"_28d_mean"))

for i in ["Trainer","Jockey"]:
    race_data[i + "_28d_AE"] = race_data["Won_"+i+"_28d_sum"]/race_data["Dec_"+i+"_28d_sum"]

numeric = list(race_data.dtypes[race_data.dtypes == 'float64'].index)
rm = [
    "Dist_F",'Dist_M',"Draw","Btn",
    "Ovr_Btn","Dec","TS","Prize",
    "Time_secs","Time_off_winner",
    "Time_off_cd_record","Time__perf_rating",
    "Btn_perf_rating",'Course_dist_record','RPR'
    ]
numeric = [i for i in numeric if i not in rm]

#Normalise numeric columns by group
def normalize_by_group(df, by, on):
    groups = df.groupby(by)[on]
    # computes group-wise mean/std,
    # then auto broadcasts to size of group chunk
    mean = groups.transform(np.mean)
    std = groups.transform(np.std)
    return (df[mean.columns] - mean) / std

race_data = race_data.merge(normalize_by_group(race_data, "raceID",numeric), left_index = True,right_index = True,suffixes = ("",'_race_std'))


none_features =["Time_secs","RPR","Time","Pos",
                "raceID","UID","Off","Name",
                "Type","Dist_F","Btn","Horse",
                "TS","Prize","Comment","Time_off_winner","Time_off_cd_record",
                "Time__perf_rating","Btn_perf_rating","Held_up",
                "Slow_into_stride","Keen","Hampered","Stayed_on","Missed_break","Weakened",
                "Raced_towards_front","Date_last_race","Comment_last_race","Date","Ovr_Btn","Dec","Won",
                'First_race', 'up_in_trip', 'down_in_trip', 'HGchange', 'new_trainer', 'new_owner',
                'Held_up_last_race', 'Slow_into_stride_last_race',
                'Keen_last_race', 'Hampered_last_race', 'Stayed_on_last_race',
                'Missed_break_last_race', 'Weakened_last_race', 'Raced_towards_front_last_race'
                ]
num_int = list(race_data.dtypes[(race_data.dtypes == 'float64')|(race_data.dtypes == 'int64') ].index)
num_int = [i for i in num_int if i not in none_features]

race_data[num_int] = race_data[num_int].apply(lambda x: (x - x.mean())/x.std())

race_data.fillna(0, inplace = True)
race_data.to_csv("/Users/leomccarthy/Documents/Projects/Horse_racing/rpscrape1/data/flat/gb/2013_2019_v1.csv", index = False)
