import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score 
from sklearn.preprocessing import StandardScaler
import imblearn
from imblearn.combine import SMOTEENN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier



# Reading the csv data in read mode
data= pd.read_csv(r"C:\Users\USER\Desktop\Customer Conversion Prediction\train.csv")
size=data.shape
# print(size)

# Checking if it is an imbalanced data
target_count=data.y.value_counts()
# print(target_count)
# it is an imbalanced data since majority class is "no"

#Cleaning
# 1.Null values
null=data.isnull().sum()
# print(null)
#  there are no null values

# 2. Duplicates 
data=data.drop_duplicates()
#  no duplicates

# 3. Outliers: To detect outliers we need to calculate the iqr. Any value >q3+(1.5*iqr) and <q1-(1.5*iqr) are outliers
outliers=data.describe()
# print(outliers)

# age
iqr_age=data.age.quantile(0.75)-data.age.quantile(0.25)
upper_threshold_age=data.age.quantile(0.75)+(1.5*iqr_age)
lower_threshold_age=data.age.quantile(0.25)-(1.5*iqr_age)
# print(upper_threshold_age) 
# print(lower_threshold_age)
# age has outlier (greater than upper_threashold).Clipping outliers
data.age=data.age.clip(lower_threshold_age,upper_threshold_age)
# print(data.age.describe())


# day
iqr_day=data.day.quantile(0.75)-data.day.quantile(0.25)
upper_threshold_day=data.day.quantile(0.75)+(1.5*iqr_day)
lower_threshold_day=data.day.quantile(0.25)-(1.5*iqr_day)
# print(upper_threshold_day) 
# print(lower_threshold_day)
# day does'nt have any outliers in it

# dur
iqr_dur=data.dur.quantile(0.75)-data.dur.quantile(0.25)
upper_threshold_dur=data.dur.quantile(0.75)+(1.5*iqr_dur)
lower_threshold_dur=data.dur.quantile(0.25)-(1.5*iqr_dur)
# print(upper_threshold_dur) 
# print(lower_threshold_dur)
# dur has an oulier(more than upper_threshold).Clipping it
data.dur=data.dur.clip(lower_threshold_dur,upper_threshold_dur)
# print(data.dur.describe())

# num_calls
iqr_numcalls=data.num_calls.quantile(0.75)-data.num_calls.quantile(0.25)
upper_threshold_numcalls=data.num_calls.quantile(0.75)+(1.5*iqr_numcalls)
lower_threshold_numcalls=data.num_calls.quantile(0.25)-(1.5*iqr_numcalls)
# print(upper_threshold_numcalls) 
# print(lower_threshold_numcalls)
# num_calls have outlier(more than upper_threshold) Clipping outliers
data.num_calls=data.num_calls.clip(lower_threshold_numcalls,upper_threshold_numcalls)
# print(data.num_calls.describe())

# 4. Format
# print(data.dtypes)
# all the data is in right format

# EDA
# for continuous columns
# plt.figure(figsize=(12,6))
# plt.subplot(2,2,1)
# plt.xlabel("age")
# plt.hist(data=data, x="age")
# plt.subplot(2,2,2)
# plt.xlabel("day")
# plt.hist(data=data, x="day")
# plt.subplot(2,2,3)
# plt.xlabel("dur")
# plt.hist(data=data, x="dur")
# plt.subplot(2,2,4)
# plt.xlabel("num_calls")
# plt.hist(data=data, x="num_calls")
# plt.tight_layout()
# plt.show()

# for categorical columns
# plt.figure(figsize=(12,6))
# df_source = data.job.value_counts()
# df_source = df_source.reset_index()
# plt.pie(df_source['job'],labels=df_source['index'],autopct='%1.1f%%')
# plt.title('Job')
# plt.axis('equal')
# plt.show()
# according to is pie chart we can see that more blue collar..........
# print(data.job.unique())

# df_source1 = data.marital.value_counts()
# df_source1 = df_source1.reset_index()
# plt.pie(df_source1['marital'],labels=df_source1['index'],autopct='%1.1f%%')
# plt.title('Marital')
# plt.axis('equal')
# # plt.show()

# df_source2 = data.education_qual.value_counts()
# df_source2 = df_source2.reset_index()
# plt.pie(df_source2['education_qual'],labels=df_source2['index'],autopct='%1.1f%%')
# plt.title('Education_qual')
# plt.axis('equal')
# # plt.show()

# df_source3 = data.call_type.value_counts()
# df_source3 = df_source3.reset_index()
# plt.pie(df_source3['call_type'],labels=df_source3['index'],autopct='%1.1f%%')
# plt.title('Call_type')
# plt.axis('equal')
# # plt.show()

# plt.figure(figsize=(10,6))
# df_source4 = data.mon.value_counts()
# df_source4 = df_source4.reset_index()
# plt.pie(df_source4['mon'],labels=df_source4['index'],autopct='%1.1f%%')
# plt.title('Mon')
# plt.axis('equal')
# # plt.show()

# df_source5 = data.prev_outcome.value_counts()
# df_source5 = df_source5.reset_index()
# plt.pie(df_source5['prev_outcome'],labels=df_source5['index'],autopct='%1.1f%%')
# plt.title('Prev_outcome')
# plt.axis('equal')
# plt.show()

# dealing with unknown values in columns
unknown_columns=["job", "education_qual"]
for i in unknown_columns:
    mode=data[i].mode()
    data[i]=data[i].replace(["unknown"], mode)

# Feature v/s Target
# for cat columns

data["New_target"]=data["y"].copy()
data["New_target"]= data["New_target"].map({"yes":1, "no":0})
cat_columns= data.select_dtypes(include=['object']).columns
# for i in cat_columns:
#     data.groupby(i)["New_target"].mean().sort_values(ascending=False).plot(kind="bar")
#     plt.show()

# for continuous columns
# cont_columns=data.select_dtypes(include=[""]
# print(data.head())

# Encoding
# Label encoding:
data["job"]=data["job"].map({'blue-collar':0, 'entrepreneur':1, 'housemaid':2,  'services':3, 'technician':4, 'self-employed':5, 'admin.':6, 'management':7, 'unemployed':8, 'retired':9, 'student':10 })
data["marital"]=data["marital"].map({'married':0, 'divorced':1, 'single':2 })
data["education_qual"]=data["education_qual"].map({ 'primary':0, 'secondary':1, 'tertiary':2})
# One hot encoding:
data=pd.get_dummies(data, columns=['call_type', 'mon', 'prev_outcome'])
# print(data.head())

# Splitting:
X=data[['age', 'job', 'marital', 'education_qual', 'day', 'dur', 'num_calls',
        'call_type_cellular', 'call_type_telephone',
       'call_type_unknown', 'mon_apr', 'mon_aug', 'mon_dec', 'mon_feb',
       'mon_jan', 'mon_jul', 'mon_jun', 'mon_mar', 'mon_may', 'mon_nov',
       'mon_oct', 'mon_sep', 'prev_outcome_failure', 'prev_outcome_other',
       'prev_outcome_success', 'prev_outcome_unknown']].values
y=data['New_target'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

# Balancing the training data:
smt = SMOTEENN(sampling_strategy='all') 
X_smt, y_smt = smt.fit_resample(X_train, y_train) 
import collections, numpy
counter = collections.Counter(y_smt)
# print(counter)


# Scaling
scaler=StandardScaler()
X_train_scaled = scaler.fit_transform(X_smt)
X_test_scaled = scaler.transform(X_test)

# Model Building
# 1.Logistic Regression
logistic_regression=LogisticRegression()
logistic_regression.fit(X_train_scaled,y_smt)
LR_pred=logistic_regression.predict(X_test_scaled)
LR_proba=logistic_regression.predict_proba(X_test_scaled)
c1=confusion_matrix(y_test,LR_pred)
# lr_plot=plot_roc_curve(logistic_regression, X_test_scaled, y_test)
# print(LR_proba)
# print(LR_pred)
# plt.show()
# print(c1)
# print(lr_plot)
#0.90 auroc

# 2. KNN
# finding the best vlue of k
# for i in [5,6,7,8,9,10,11,12,13,14,15]:
#     knn=KNeighborsClassifier(i)
#     knn.fit(X_train_scaled,y_smt)
#     print("k value :" ,i, "trainscore :", knn.score(X_train_scaled, y_smt), "cv score :", np.mean(cross_val_score(knn, X_train_scaled, y_smt, cv=10, scoring="roc_auc")))
knn=KNeighborsClassifier(8)
knn.fit(X_train_scaled,y_smt)
knn_pred=knn.predict(X_test_scaled)
# knn_plot=plot_roc_curve(knn, X_test_scaled, y_test)
# print(knn_plot)
# plt.show()
# print(knn_pred)
# 0.88

# 3.Decision tree
dt = DecisionTreeClassifier()
# dt.fit(X_train_scaled,y_smt)
# dt_pred=dt.predict(X_test_scaled)
# dt_plot=plot_roc_curve(dt,X_test_scaled, y_test)
# print(dt_plot)
# plt.show()
# doing cross validation
# for depth in [10,15,16,17,18,19,20,21,22]:
#     dt = DecisionTreeClassifier(max_depth=depth)
#     dt.fit(X_train_scaled, y_smt)
#     trainAccuracy = accuracy_score(y_smt, dt.predict(X_train_scaled))
#     valAccuracy = cross_val_score(dt, X_train_scaled, y_smt, cv=10)
#     print("Depth  : ", depth, " Training Accuracy : ", trainAccuracy, " Cross val score : " ,np.mean(valAccuracy))
    
dt = DecisionTreeClassifier(max_depth=17)
dt.fit(X_train_scaled, y_smt)
# dt_plot=plot_roc_curve(dt,X_test_scaled, y_test)
# print(dt_plot)
# plt.show()
# 0.77


# 4.Xgboost
# for lr in [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.2,0.5,0.7,1]:
#       model = xgb.XGBClassifier(learning_rate = lr, n_estimators=100, verbosity = 0)  
#       print("Learning rate : ", lr," Cross-Val score : ", np.mean(cross_val_score(model, X_test_scaled, y_test, cv=10)))

model = xgb.XGBClassifier(learning_rate = 0.2, n_estimators=100)     
model.fit(X_train_scaled,y_smt)
# xgb_plot=plot_roc_curve(model,X_test_scaled,y_test)
# print(xgb_plot)
# plt.show()
# 0.92

# 5.Random forest
rf=RandomForestClassifier(n_estimators=100,criterion="entropy")
rf.fit(X_train_scaled,y_smt)
rf_plot=plot_roc_curve(rf,X_test_scaled,y_test)
print(rf_plot)
plt.show()
# 0.92