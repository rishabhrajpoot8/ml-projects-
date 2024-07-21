#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import seaborn as sns
import pickle
import re


# In[ ]:


print('hi therre')


# In[ ]:


csvfile = '/content/water_potability (1).csv'       # while reading the data the compiler was not able to decode the utf-8 so we changed the encoding
data_all = pd.read_csv(csvfile , encoding = 'latin-1' )


# 

# In[ ]:


data_all.head()     # this is how our data looks like


# In[ ]:





# In[ ]:


data_all.info()


# In[ ]:


data_all.shape   # shape of our data


# In[ ]:


data_all['Potability'].value_counts()    # potability is our target column


# In[ ]:


338/data_all.shape[0]                  # it has class imbalance of around 90 /10


# In[ ]:


# EDA
data_all.head()                  # if we look at the data we can see that all the columns should numeric in nature


# In[ ]:


data_all.info()        # ph , hardness , solids , chloramines  , conductivity , organic carbon ,Trihalomethanes and turbidity should be numeric in nature but
                       # there should be some probllem with it as it is coming out to object column


# In[ ]:


data_all.isna().sum()  #as we can see there are 3 columns which contains nul values
                        # ph , sulphate and trihalomethanes
                        # we have to do null value treatment


# In[ ]:


# preprocessing
# we have to do null value treatment
# find the values which are causing issues with the numeric column and change it into object column



# In[ ]:


def non_numeric(column):
  non_numeric_elements = []
  for i in column:
    try:
      float(i)
    except:
      non_numeric_elements.append(i)
  return non_numeric_elements


# In[ ]:


column_list = ['ph','Hardness','Chloramines','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']

for col in column_list:
  non_numeric_elements1 = non_numeric(data_all[col])
  print('*********')
  print(col)
  print(non_numeric_elements1)
  print('no of values' , len(non_numeric_elements1))


# there are many vlues which includes special character and string values we have to treat it


# In[ ]:


data_all.loc[data_all['ph']=='7.-.160467231','ph'] = 7.160467231
data_all.loc[data_all['Hardness']=='214.496610%457156','Hardness'] = 214.496610457156
data_all.loc[data_all['Hardness']=="20''9.609618",'Hardness'] = 209.609618


# we have treated the oject values of ph and hardness



# In[ ]:


data_all.head()


# In[ ]:


def object_to_float(s):
  ''' This script defines a function called convert_to_float, which first cleans the string by using re.sub/
   to remove any characters other than digits or dots, and then transforms it back to a float. '''

  empty_string = ''

    # Attempt to clean the string
  try:
      empty_string = re.sub(r'[^\d.]', '', s)
  except Exception as e:
        print(f"Error cleaning string: {e}")

  return float(empty_string) if empty_string else None

#Apply the conversion function to the specified column
column_list = ['Hardness','Chloramines','Conductivity','Organic_carbon','Turbidity']
for col in column_list:
  data_all[col] = data_all[col].apply(lambda x: object_to_float(x))
  print('******')
  print(col)
  print(data_all[col])


# In[ ]:


data_all.info()


# In[ ]:


def convert_to_float(s):
    if pd.isnull(s) or not isinstance(s, str):
        return None

    # Remove non-digit and non-dot characters
    cleaned_string = re.sub(r'[^\d.]', '', s)

    #print(f"Original: {s}, Cleaned: {cleaned_string}")

    # Convert to float
    try:
        return float(cleaned_string)
    except ValueError:
        return None
data_column = ['Solids','ph','Trihalomethanes']
for col in data_column:
# Apply the conversion function to each column in data_column

  data_all[col] = data_all[col].apply(lambda x: convert_to_float(x))
#data_all['Solids'] = data_all['Solids'].apply(lambda x: convert_to_float(x))

# Print the DataFrame with converted values
  print(data_all[col])
  print('**********************************')


# In[ ]:


data_all.info()                 # changed to datatype of columns object to float


# In[ ]:


# null value treatement
data_all.isnull().sum()


# In[ ]:


data_all['ph'].describe()


# In[ ]:





# In[ ]:





# In[ ]:


# checking skewness
data_all.skew()                # 3 features has skewness majorly chloramines organic_carbon and solids


# In[ ]:


sns.distplot(data_all['Solids'])


# In[ ]:


sns.distplot(data_all['Chloramines'])


# In[ ]:


sns.distplot(data_all['Organic_carbon'])


# In[ ]:


sns.heatmap(data_all[column_list1].corr())


# In[ ]:


data_all[column_list1].corr()


# In[ ]:


column_list1=['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']


# In[ ]:


data_all.loc[data_all['ph'].isna(),'ph'] = np.mean(data_all['ph'])
   # changing the null to mean values bcz data is almost normally distributed


# In[ ]:


data_all.loc[data_all['Hardness'].isna(), 'Hardness'] = np.mean(data_all['Hardness'])
     # changing the null to mean values bcz data is almost normally distributed


# In[ ]:


data_all['Solids'].describe()


# In[ ]:


data_all.loc[data_all['Solids'].isna(),'Solids'] =21200
    # changing the null value to 21200 because there is siome little skewness


# In[ ]:


data_all.loc[data_all['Sulfate'].isna(),'Sulfate'] = np.mean(data_all['Sulfate'])
      # changing the null to mean values bcz data is almost normally distributed


# In[ ]:


data_all.loc[data_all['Trihalomethanes'].isna(),'Trihalomethanes'] = np.mean(data_all['Trihalomethanes'])
      # changing the null to mean values bcz data is almost normally distributed


# In[ ]:


data_all.info()
 # all the null value has been treated


# In[ ]:


data_all.skew()


# In[ ]:


from matplotlib import pyplot as plt

numeric_cols = data_all.select_dtypes(include = np.number) ### selects numeric columns

column_names = list(numeric_cols.columns)

col_index = 0

plot_rows = 5
plot_cols = 2

fig, ax = plt.subplots(nrows = plot_rows,ncols=plot_cols,figsize = (20,20))

for row_count in range(plot_rows):
    for col_count in range(plot_cols):
        ax[row_count][col_count].scatter(y = numeric_cols[column_names[col_index]],x=numeric_cols.index)
        ax[row_count][col_count].set_ylabel(column_names[col_index])
        col_index = col_index + 1


# In[ ]:


# two columns has outliers
#1 organic_carbon
#2 chloramines
# as we have seen above these two columns were right skewed also
# and the needs to be scaled also


# In[ ]:


#outliers treatment
data_all.loc[data_all['Organic_carbon']>2000 , 'Organic_carbon'] = 16.560201   #changing it to 75th percentile


# In[ ]:


data_all['Organic_carbon'].describe()


# In[ ]:


data_all.loc[data_all['Chloramines']>2000 , 'Chloramines'] = 8.115458  #changing it to 75th percentile


# In[ ]:


data_all['Chloramines'].describe()


# In[ ]:


data_all.skew()
                                             # there is little bit skewness in some columns , lets treat in next iteration


# In[ ]:


data_all.head(20)        # the data is quite sorted now


# In[ ]:


sns.distplot(data_all['Solids'])


# In[ ]:


sns.distplot(np.sqrt(data_all['Solids']))


# In[ ]:


sns.distplot(data_all['Organic_carbon'])


# In[ ]:


sns.heatmap(numeric_cols.corr())


# In[ ]:


np.sqrt(data_all['Solids']).skew()


# In[ ]:


data_all['Solids'] = np.sqrt(data_all['Solids'])


# In[ ]:


sns.distplot(data_all['Conductivity'])


# In[ ]:


sns.distplot(np.sqrt(data_all['Conductivity']))


# In[ ]:


np.sqrt(data_all['Conductivity']).skew()


# In[ ]:


data_all['Conductivity'] = np.sqrt(data_all['Conductivity'])



# In[ ]:


sns.distplot(np.sqrt(data_all['Organic_carbon']))


# In[ ]:


data_all.skew()


# In[ ]:


# checking correlation
data_all.corr()


# In[ ]:





# In[ ]:


#scaling
from sklearn.preprocessing import StandardScaler
column_names.remove('Potability')
sc = StandardScaler()
sc.fit(data_all[column_names])
data_all[column_names] = sc.transform(data_all[column_names])


# In[ ]:


feature_columns = data_all.drop(['Potability'] , axis =1 )
target_column = data_all['Potability']


# In[ ]:


feature_columns.shape , target_column.shape


# In[ ]:


# modelling
# first divide the data into trainin and testing data


# In[ ]:


from sklearn.model_selection import train_test_split
train_feature, test_feature , train_target , test_target = train_test_split(feature_columns,target_column,\
test_size = 0.2 , random_state = 1200 , stratify= target_column)


# In[ ]:


train_feature.shape, test_feature.shape , train_target.shape , test_target.shape


# In[ ]:


train_target.value_counts()


# In[ ]:


test_target.value_counts()   # WE can see there is class imbalance in the data


# In[ ]:


train_feature.reset_index(drop = True, inplace = True)
test_feature.reset_index(drop = True, inplace = True)
train_target.reset_index(drop = True, inplace = True)
test_target.reset_index(drop = True, inplace = True)


# In[ ]:


# first iteration of logistic regression
from sklearn.linear_model import LogisticRegression

LR_model = LogisticRegression(max_iter = 500,
                              random_state = 8,
                              class_weight = {0:1,1:4})
LR_model.fit(train_feature,train_target)
# without scaling and skewness treatment
# with threshod of 0.3 ,  class_weight = {0:1,1:4} f1 score 18.9
#                         class_weight = {0:1,1:5} f1 score train is 18.3 and f1 score of test is improved from 0 to 17.5
#                         class_weight = {0:1,1:5.9} f1 score train is 18.69 and f1 score of test is improved from 17.5 to 18.62.
# with scaling skewness treatment
#                         class_weight = {0:1,1:5.9} f1 score train is 18.69 and f1 score of test is  18.62. no change


# In[ ]:


Prediction = pd.DataFrame(LR_model.predict_proba(train_feature))


# In[ ]:


Prediction['Threshold'] = 0
Prediction.loc[Prediction[1]>=0.3,'Threshold'] = 1


# In[ ]:


Prediction


# In[ ]:


from sklearn.metrics import f1_score , confusion_matrix
f1_score(y_true=train_target, y_pred = Prediction['Threshold'])


# In[ ]:


pd.DataFrame(confusion_matrix(y_true = train_target, y_pred = Prediction['Threshold'] ) , columns = ['Predict_0','Predict_1'] , index = ['Actual_0','Actual_1'])


# In[ ]:


prediction_test = pd.DataFrame(LR_model.predict_proba(test_feature))


# In[ ]:


prediction_test['Threshold']=0
prediction_test.loc[prediction_test[1]>=0.4,'Threshold'] = 1


# In[ ]:


prediction_test


# In[ ]:


f1_score(y_true=test_target, y_pred = prediction_test['Threshold'])


# In[ ]:


train_rows = []
test_rows = []

from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=3) ## divide the entire data into 5 equal parts..each part is of 20% data

splits = kf.split(X=feature_columns, y=target_column) ## actual splitting happens

for train_index, test_index in splits:
  train_rows.append(list(train_index))
  test_rows.append(list(test_index))


# In[ ]:


train_data = []
test_data = []

for train_index, test_index in zip(train_rows, test_rows):
    X_train, X_test = feature_columns.iloc[train_index], feature_columns.iloc[test_index]
    y_train, y_test = target_column.iloc[train_index], target_column.iloc[test_index]

    train_data.append((X_train, y_train))
    test_data.append((X_test, y_test))


# In[ ]:


first_fold_train_features, first_fold_train_target = train_data[0]
first_fold_val_features, first_fold_val_target = test_data[0]

Second_fold_train_features, Second_fold_train_target = train_data[1]
Second_fold_val_features, Second_fold_val_target = test_data[1]

third_fold_train_features, third_fold_train_target = train_data[2]
third_fold_val_features, third_fold_val_target = test_data[2]




# In[ ]:


LR_model = LogisticRegression(max_iter = 400, random_state = 800,
                              class_weight = {0:1,1:5.6})
LR_model.fit(first_fold_train_features,first_fold_train_target)


# In[ ]:


Prediction = pd.DataFrame(LR_model.predict_proba(first_fold_train_features))


# In[ ]:


Prediction['Threshold'] = 0
Prediction.loc[Prediction[1]>=0.4,'Threshold'] = 1


# In[ ]:


f1_score(y_true=first_fold_train_target, y_pred = Prediction['Threshold'])


# In[ ]:


prediction_test = pd.DataFrame(LR_model.predict_proba(first_fold_val_features))


# In[ ]:


prediction_test['Threshold'] = 0
prediction_test.loc[prediction_test[1]>=0.4,'Threshold'] = 1


# In[ ]:


f1_score(y_true=first_fold_val_target, y_pred = prediction_test['Threshold'])


# In[ ]:


LR_model = LogisticRegression(max_iter = 400, random_state = 1200,
                              class_weight = {0:1,1:5.6})
LR_model.fit(Second_fold_train_features,Second_fold_train_target)


# In[ ]:


prediction_train = pd.DataFrame(LR_model.predict_proba(Second_fold_train_features))
prediction_val = pd.DataFrame(LR_model.predict_proba(Second_fold_val_features))


# In[ ]:


prediction_train['Thres'] = 0
prediction_train.loc[prediction_train[1]>=0.4,'Thres'] = 1
prediction_val['Thres'] = 0
prediction_val.loc[prediction_val[1]>=0.4,'Thres'] = 1


# In[ ]:


print(f"f1 score for Second_fold_train {f1_score(y_true= Second_fold_train_target, y_pred = prediction_train['Thres'])}")
print(f"f1 score for Second_fold_val {f1_score(y_true= Second_fold_val_target, y_pred = prediction_val['Thres'])}")


# In[ ]:


LR_model = LogisticRegression(max_iter = 400, random_state = 1200,
                              class_weight = {0:1,1:5.6})
LR_model.fit(third_fold_train_features,third_fold_train_target)


# In[ ]:


# decision tree


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(  criterion='entropy' ,
                              class_weight={0:1,1:6},
                             max_depth = 6, min_samples_split = 10 , random_state=1200)
dtc.fit(train_feature,train_target)
#                       class_weight={0:1,1:5} with this f1 score train 43.3 and test is 20.1
#                         criterion='entropy'  with this f1 score train 43.3 and test is 16
#                       class_weight={0:1,1:6} with this f1 score train 43.3 and test is 15.6
#                         criterion='entropy'  with this f1 score train 43.3 and test is 16.5         conclusion = it dint get better at all
#                       now with threshold 0.4 and class_weight={0:1,1:5}f1 score train 31.5 and test is 21.5
#
#                       now with threshold 0.3 and class_weight={0:1,1:5}f1 score train 31.4 and test is 20.7
#                       now with threshold 0.4 and class_weight={0:1,1:6}f1 score train 31.2 and test is 20.9
#                       after using grid search cv max value f1 train test score is  55.9 and test is 20.0


# In[ ]:


prediction = pd.DataFrame(dtc.predict(train_feature))


# In[ ]:


prediction


# In[ ]:


pd.DataFrame(confusion_matrix(y_true = train_target , y_pred = prediction) ,columns = ['Predict_0','Predict_1'] , index = ['Actual_0','Actual_1'])


# In[ ]:


f1_score(y_true = train_target, y_pred = prediction)


# In[ ]:


predict_test = pd.DataFrame(dtc.predict(test_feature))


# In[ ]:


f1_score(y_true = test_target, y_pred = predict_test )


# In[ ]:


predict_threshold = pd.DataFrame(dtc.predict_proba(train_feature))


# In[ ]:


predict_threshold['Threshold'] = 0
predict_threshold.loc[predict_threshold[1]>=0.3,'Threshold'] = 1


# In[ ]:


f1_score(y_true = train_target, y_pred = predict_threshold['Threshold'] )


# In[ ]:


predict_threshold_test = pd.DataFrame(dtc.predict_proba(test_feature))


# In[ ]:


predict_threshold_test['Threshold'] = 0
predict_threshold_test.loc[predict_threshold_test[1]>=0.3,'Threshold'] = 1


# In[ ]:


f1_score(y_true = test_target, y_pred = predict_threshold_test['Threshold'] )


# In[ ]:


from sklearn.model_selection import GridSearchCV
dtc = DecisionTreeClassifier()
params = { 'class_weight' : [{0:1,1:4.5},{0:1,1:5},{0:1,1:5.1},{0:1,1:5.3}],
          'max_depth' : [6,7,8,9,10],
           'min_samples_split' : [10,20,25]

           }
grid_search = GridSearchCV(estimator = dtc,           #class
                           param_grid = params,
                           cv=5,           #Kfold validation = 5 fold validation
                           scoring = 'f1',
                           return_train_score=True )
grid_search.fit(feature_columns,target_column)


# In[ ]:


grid_search.cv_results_['mean_train_score']


# In[ ]:


grid_search.cv_results_['mean_test_score']


# In[ ]:


#ensemble technique


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

lr_model =  LogisticRegression(max_iter=500)
svm_model = SVC()
knn_model = KNeighborsClassifier()
rf_model = RandomForestClassifier()
dt_model = DecisionTreeClassifier()

ensemble_model= VotingClassifier([('lr', lr_model),
                                  ('svm', svm_model),
                                  ('knn', knn_model),
                                  ('rf', rf_model),
                                  ('dt', dt_model)], voting='hard')

ensemble_model.fit(feature_columns, target_column)


# In[ ]:


from sklearn.metrics import confusion_matrix
prediction = ensemble_model.predict(feature_columns)
actuals = target_column.tolist()



# In[ ]:


confusion_matrix(actuals, prediction)


# In[ ]:


f1_score(actuals, prediction)


# In[ ]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(train_feature, train_target.ravel())




# In[ ]:


print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))


# In[ ]:


LR_model = LogisticRegression(max_iter = 400,                                  ## Model is overfitting with LR tried with 40%, 50%
                              class_weight = {0:1,1:1})
LR_model.fit(X_train_res,y_train_res)
#                        after using smote
#                                     the train  f1 score at {0:1,1:2} is 66.6 test is 18.7
#                                     the train  f1 score at {0:1,1:2} is 54.4 test is 20.1    # there was no change after channging the threshold also


# In[ ]:


Prediction_smote = pd.DataFrame(LR_model.predict_proba(X_train_res))


# In[ ]:


Prediction_smote['threshold']=0
Prediction_smote.loc[Prediction_smote[1]>=0.4,'threshold'] = 1


# In[ ]:


Prediction_smote


# In[ ]:


f1_score(y_true = y_train_res , y_pred =Prediction_smote['threshold'] )


# In[ ]:


Prediction_test = pd.DataFrame(LR_model.predict_proba(test_feature))


# In[ ]:


Prediction_test['threshold']=0
Prediction_test.loc[Prediction_test[1]>=0.4,'threshold'] = 1


# In[ ]:


Prediction_test


# In[ ]:


f1_score(y_true = test_target , y_pred =Prediction_test['threshold'] )


# In[ ]:


clf = DecisionTreeClassifier(class_weight={0:1,1:1},
                             max_depth = 8, min_samples_split = 10 , random_state=8)
clf.fit(X_train_res,y_train_res)
#                            at threshold 0.3 and class_weight={0:1,1:3} test f1 score is 73.03 and test is 18.9
#                            at threshold 0.3 and class_weight={0:1,1:3} test f1 score is 80.5 and test is 19.6
#                            no change in test f1 score after varying max and everything although train f1 score reached to 75.5 at max depth 8
#                            and it was 75.6 at threshold of .5


# In[ ]:


predict = pd.DataFrame(clf.predict_proba(X_train_res))


# In[ ]:


predict['threshold'] = 0
predict.loc[predict[1]>=0.5,'threshold'] = 1


# In[ ]:


predict


# In[ ]:


print(f1_score(y_true= y_train_res, y_pred= predict['threshold']))


# In[ ]:


predict_test = pd.DataFrame(clf.predict_proba(test_feature))


# In[ ]:


predict_test


# In[ ]:


predict_test['threshold'] = 0
predict_test.loc[predict[1]>=0.3,'threshold'] = 1


# In[ ]:


print(f1_score(y_true= test_target, y_pred= predict_test['threshold']))

