import pandas as pd
import lightgbm as lgb
import numpy as np
import shap
from sklearn.metrics import log_loss
import tensorflow as tf
from tensorflow import keras
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV

race_data = pd.read_csv("rpscrape1/data/flat/gb/2013_2019_v1.csv")
race_data["Date"] = race_data["Date"].astype('datetime64')


race_data.drop(race_data[race_data["Date"].dt.year < 2014].index, inplace = True)


none_features =["Time_secs","RPR","Time","Pos",
                "UID","Off","Name",
                "Type","Dist_F","Btn","Horse",
                "TS","Prize","Comment","Time_off_winner","Time_off_cd_record",
                "Time__perf_rating","Btn_perf_rating","Held_up",
                "Slow_into_stride","Keen","Hampered","Stayed_on","Missed_break","Weakened",
                "Raced_towards_front","Date_last_race","Comment_last_race","Date","Ovr_Btn",
                "Owner","Jockey","Trainer","Owner_last_race","Trainer_last_race",'Sire', 'Dam', 'Damsire',
                "Course",'Course_last_race', 'Rating_Band_last_race','Track_speed_last_race',"raceID","Dec"

                ]

#Some one hot encoding categorical features
race_data["Track_direction"] = np.where(race_data["Track_direction"] == "L", 0,1)
race_data['Track_speed'] = pd.factorize(race_data['Track_speed'])[0] + 1
cat = list(race_data.drop(none_features,axis = 1).dtypes[race_data.drop(none_features, axis =1).dtypes == "object"].index)
race_data = pd.concat([race_data,pd.get_dummies(race_data[cat], prefix=cat).astype("float64")],axis=1)
race_data.drop(cat,axis=1, inplace=True)


#Find none numeric columns
# cat_features = list(race_data.drop(none_features,axis = 1).dtypes[race_data.drop(none_features, axis =1).dtypes == "object"].index)
# race_data[cat_features] = race_data[cat_features].astype("category")

train  = race_data.loc[race_data["Date"] <= pd.to_datetime('2018-04-30'),].drop(none_features, axis =1)
val = race_data.loc[(race_data["Date"].dt.year < 2019 ) & (race_data["Date"] > pd.to_datetime('2018-04-30')),].drop(none_features, axis =1)
test = race_data.loc[race_data["Date"].dt.year == 2019,].drop(none_features, axis =1)


target = "Won"
X_train = train.drop(target, axis = 1)
y_train = train[target]
X_val = val.drop(target, axis = 1)
y_val = val[target]
X_test = test.drop(target, axis = 1)
y_test = test[target]




#
train_lgb = lgb.Dataset(X_train,label = y_train,free_raw_data=False)
val_lgb = lgb.Dataset(X_val,label = y_val, reference  = train_lgb,free_raw_data=False)
test_lgb = lgb.Dataset(X_test,label = y_test,free_raw_data=False)
#
# param_grid = {
#     'num_leaves': [6,12,18,24,32,64],
#      'eta': [0.01,0.05,0.1,0.2],
#      'feature_fraction':[0.2,0.4,0.6,0.8,1],
#      'bagging_fraction':[0.2,0.4,0.6,0.8,1],
#      'bagging_freq': [0,1,4,6,8],
#      'max_depth': [3,6,9,12,16,-1],
#      'reg_lambda' :[0,0.02,0.06,0.2,0.3,0.5]
#      }

param = {
    'num_leaves': 3,
    'objective': 'binary',
    'num_threads': 7,
    'eta': 0.01,
    'feature_fraction': 0.5,
    'lambda_l2': 0.01,
    'bagging_fraction': 0.5,
    'bagging_freq': 9,
    'max_depth': 6
    }
param['metric'] = 'binary_logloss'

bst = lgb.train(train_set = train_lgb,valid_sets=[val_lgb],params = param,num_boost_round=5000,verbose_eval = True, early_stopping_rounds = 30)



explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(X_test)

shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0][62,:],X_test.iloc[62,:], matplotlib = True)
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.summary_plot(shap_values, X_val)


clfLGB  = lgb.LGBMClassifier(n_estimators = 5000,**param)
clfLGB.fit(X_train,y_train)

clfNN = keras.Sequential([
    keras.layers.Flatten(input_shape=(400,)),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dropout(0.4),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

clfNN.compile(optimizer= tf.keras.optimizers.Adam(0.001),
              loss='binary_crossentropy',
              metrics=["binary_crossentropy"])

clfNN.fit(X_train, y_train, epochs=15, batch_size=20,validation_data = (X_val,y_val))

clfNN.evaluate(X_test, y_test)


probs_lgb = bst.predict(X_val)
probs_nn = clfNN.predict(X_val)

valid = pd.DataFrame({"lgb": probs_lgb,"Won":y_val,"SP":1/SP , "raceID":race_data.loc[X_val.index,"raceID"]})
test = pd.DataFrame({"lgb": t_lgb,"Won":y_test,"SP":1/t_sp , "raceID":race_data.loc[X_test.index,"raceID"]})
test.to_csv("clogit_test.csv")
valid.to_csv("clogit_val.csv")

fop, mpv = calibration_curve(y_test,calibrator.predict_proba(X_test)[:,1], n_bins=10)
# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--')
# plot model reliability
plt.plot(mpv, fop, marker='.')
plt.show()




SP = race_data.loc[X_val.index,"Dec"]

X_stack = pd.DataFrame({"lgb": probs_lgb, "nn":probs_nn.flatten()})

from sklearn.svm import SVC
clf_stack = SVC(gamma='auto')
clf_stack.fit(X_stack,y_val)

t_lgb = bst.predict(X_test)
t_nn = clfNN.predict(X_test)
t_sp = race_data.loc[X_test.index,"Dec"]
X_t = pd.DataFrame({"lgb": t_lgb, "nn" : t_nn.flatten()})

log_loss(y_test,clf_stack.predict(X_t))



shap_sum = np.abs(shap_values[1]).mean(axis = 0)
importance_df = pd.DataFrame([X_val.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['column_name', 'shap_importance']
importance_df = importance_df.sort_values('shap_importance', ascending=False)
imp = list(importance_df.iloc[0:400,0])
importance_df.iloc[0:400,]



race_test = race_data.loc[X_test.index,]
race_test['probs'] = bst.predict(X_test)

race_test["odds"] = 1/race_test["probs"]
race_test['kelly'] = ((race_test["probs"]*(race_test["Dec"]) )- 1) / race_data["Dec"]
race_test["kelly"] = race_test['kelly'].clip(0,1)
race_test["bet"]  = np.where((race_test["kelly"] > 0) & (race_test["probs"] > 0.2),1,0)
race_test["earn"] = (race_test["kelly"] * 1000) * race_test["Dec"]  * race_test["Won"] * race_test["bet"]
race_test["cum_earn"] = race_test["earn"].cumsum()
race_test["cum_bet"] = (race_test["bet"] * race_test["kelly"] * 1000).cumsum()
race_test["profit"] = race_test["cum_earn"] - race_test["cum_bet"]
race_test["profit"].min()
plt.plot(race_test["cum_earn"] - race_test["cum_bet"])
race_test.loc[:,["probs","odds","odd_probs","Dec","raceID","bet","Pos","kelly","earn", "cum_earn","cum_bet","profit"]]
