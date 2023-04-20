import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StandardScaler




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

# X=data