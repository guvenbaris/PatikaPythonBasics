#!/usr/bin/env python
# coding: utf-8

# **Bu not defteri patika eğitim platformunda bulunan veri bilimi patikasının veri bilimi 101 dersinin son projesi olarak hazırlanmıştır.**

# # 1. Introduction

# **Competition Description**
# 
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.
# 
# **Practice Skills**
# * Creative feature engineering 
# * Advanced regression techniques like random forest and gradient boosting
# 
# **Acknowledgments**
# 
# The Ames Housing dataset was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset. 

# # 2. Imports 

# We import necessary libraries our notebook

# In[480]:


import os 
import math
import numpy as np 
import pandas as pd 
from zipfile import ZipFile
import matplotlib.pyplot as plt

from kaggle.api.kaggle_api_extended import KaggleApi

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error,mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR


# # 3. Load Dataset From The Kaggle

# This section we will download dataset from kaggle. 
# For this you can look in here: 
# https://medium.com/@jeff.daniel77/accessing-the-kaggle-com-api-with-jupyter-notebook-on-windows-d6f330bc6953

# In[4]:


# Connect to Kaggle with API
# Download json file from your kaggle account 
# Go to directory — “C:\Users\<username>\.kaggle\” — and paste here downloaded JSON file
api = KaggleApi() 
api.authenticate() 


# In[5]:


# download dataset from kaggle
get_ipython().system('kaggle competitions download -c house-prices-advanced-regression-techniques')


# In[6]:


# Extract the zip file to the folder under the data folder
zf = ZipFile('house-prices-advanced-regression-techniques.zip')
zf.extractall()
zf.close()


# # 4. Read Dataset

# Next step we will read dataset from file. We have two dataset by train and test.

# In[2]:


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


# In[3]:


train_df.head()


# In[4]:


test_df.head()


# # 5. Information About Dataset

# In this section, we will take a quick look at the information about the data set.

# In[37]:


# information about dataset 
# Show non-null count and dtype
train_df.info()


# In[38]:


# test df information
# Show non-null count and dtype
test_df.info()


# In[39]:


# Correlation with SalePrice
train_df.corr()["SalePrice"].sort_values(ascending = False)


# In[40]:


# train data columns number,mean ,std etc.
train_df.describe()


# In[41]:


test_df.describe()


# # 6. Preprocessing

# In this section we will prepare our train and test dataset for training. Firstly we will look NaN values. After we will decide to what can we do for these NaN values. We will scale  our numerical features so our model can do a better job. 

# <ins>**NaN Values Analyze:**</ins> 
# 
# **__Train dataset__** 
# 
# The values stated here are for the train dataset
# 
#  We have total 1460 row, some of columns has 1453, 1406, 1369, 1179, 690 NaN values so we decided  won't take training these columns. For the rest we will fill in with mostly used value.
# 
# **These columns:**
# 
#      PoolQC 1453 NaN Values
#      MiscFeature 1406 NaN Values
#      Alley 1369 NaN Values
#      Fence 1179 NaN Values
#      FireplaceQu 690 NaN Values 
# 
#  We saw that the MasVnrType column normally has a value of 8 NaN, but in reality MasVnrType column has 864 None value it wasn't show when we searched null value because "None" was write on that column because this is string type value so i decided to drop this column.
# 
# **MasVnrType Column Value Counts:**
# 
#      None       864
#      BrkFace    445
#      Stone      128
#      BrkCmn      15
# 
# Also we need drop Id column because column has unique values. Our model can be memorize them. This may mislead us in evaluating the model.
# 
# **Test dataset**
# 
# We have to drop the columns we dropped in the train data in the test. So we can evaluate our model.

# * Next step, we will define function. This function purpose is take column name but by numerical and categorical features separately.

# In[5]:


# Take column name 
def take_column(df):
    cat_dict = {}
    num_dict = {}
    for cat in df.columns:
        if df[cat].dtype == 'O':
            cat_dict[cat] = df[cat]
        else: 
            num_dict[cat] = df[cat]
    return cat_dict,num_dict


# * Next step, we will define function. This function purpose is getting column names with NaN values but by numerical and categorical features separetely. We will take column names to variable.

# In[6]:


def check_nan_values(df):
    cat_dict_nan = {}
    num_dict_nan = {}
    
    cat_col,num_col = take_column(df)
    
    for col in cat_col:
        if df.isnull().sum()[col]:
            cat_dict_nan[col] = df.isnull().sum()[col]
            
    for col in num_col:
        if df.isnull().sum()[col]:
            num_dict_nan[col] = df.isnull().sum()[col]
            
    return cat_dict_nan,num_dict_nan


# In[7]:


train_cat_nan , train_num_nan = check_nan_values(train_df)
test_cat_nan,test_num_nan = check_nan_values(test_df)


# * Next step, function will show NaN values and column name and number of NaN values. 

# In[13]:


# Let's look NaN values
def nan_values_print(nan_dict):
    """   
    Dictionary should be given. This function purpose is just for print 
    """
    print("Categorical Variables With Null Values In The Training Set\n")
    print("Columns Name \tNumbers Of NaN")
    print("-"*30)
    for i in range(len(nan_dict)):
        key = list(nan_dict.keys())[i]
        value = list(nan_dict.values())[i]
        print(f"{key}   \t|   {value}")


# In[14]:


nan_values_print(train_cat_nan)


# In[15]:


nan_values_print(train_num_nan)


# In[16]:


nan_values_print(test_cat_nan)


# In[17]:


nan_values_print(test_num_nan)


# * Next step, we decided to remove these columns, for the reason we explained above

# In[18]:


# These column we want to drop them 
col_to_drop = ["Id","PoolQC","MiscFeature","Alley","Fence","MasVnrType","FireplaceQu"]


# * Next step, we will show the number of values of categorical columns that have NaN values.

# In[19]:


# Number of values of categorical columns
def value_count_print(df):

    cat_dict, _ = check_nan_values(df)
    
    # Print categorical data with NaN values to the screen
    for cat in cat_dict.keys():
        print("*"*40)
        print(f"{cat}: \nNaN Values Numbers: {df[cat].isnull().sum()}")
        print(f"{df[cat].value_counts().sort_values(ascending=False)}\n")


# In[20]:


value_count_print(train_df)


# In[21]:


value_count_print(test_df)


# * Next step, we will define two function first function purpose is fill in the blank values. For categorical features, we will fill in that column with the most used values, for numerical features we will fill in with column mean. 
# 
# 
# * Second function purpose is preprocessing of dataset for training. We can want it to drop column, use encoder and scaler,
# and we can want it fill in the blank values. For fill the blank values it will use first function. 

# In[22]:


def fill_nan_values(df): 
    """
    This func fill nan values. 
    For categorical feature we will fill NaN values with that column mostly used value.
    For numerical feature we will fill NaN values with that column median
    Will return df with no NaN value
    """
    cat_dict, num_dict = check_nan_values(df)
    
    for cat_col in cat_dict.keys():     
        val = list(df[cat_col].value_counts().sort_values(ascending=False).to_dict().keys())[0]
        df[cat_col] = df[cat_col].fillna(value = val)
    
    for num_col in num_dict.keys():
        df[num_col] = df[num_col].fillna(df[num_col].mean())
    
    return df

def preprocess(df,drop_col=None,encoder=None,scaler=None,nan_values = False):
    """
    preprocess input data:
    param: df => Dataframe with data
    param: encoder => encoder object with fit_transform method,
    param: scaler => scaler object with fit_transform method,
    param: nan_values => Whether there is a column with NaN
    return df
    """
    if drop_col:
        df = df.drop(col_to_drop,axis = 1) 
        
    cat_columns,num_columns = take_column(df)
    
    if encoder: 
        for cat in cat_columns:
            df[cat] = encoder.fit_transform(df[cat])
    if scaler: 
        num_columns = list(num_columns)
        for col in num_columns:
            if col == 'SalePrice': # drop our label column we don't want to use scaling on.
                num_columns.remove('SalePrice')
            else:
                pass
        df[num_columns] = scaler.fit_transform(df[num_columns])
    if nan_values:
        df = fill_nan_values(df)
            
    return df


# In[27]:


# preprocessed datasets
train_data = preprocess(train_df,col_to_drop,LabelEncoder(),StandardScaler(),nan_values = True)
test_data = preprocess(test_df,col_to_drop,LabelEncoder(),StandardScaler(),nan_values = True)


# * Next step we split train dataset by train and validation. We split dataset for the real life success. We will be able to see how it will behave on a data set that it does not see. But for the submission we train all dataset because more data more success.

# In[413]:


def split_dataset(X,Y,rate= 0.20):
    
    x_train,x_val,y_train,y_val = train_test_split(X,Y,test_size=rate,random_state=1)
    
    return x_train,x_val,y_train,y_val 

X = train_data.iloc[:,:-1].values # features
y = train_data.iloc[:,-1:].values.ravel() # label

x_train,x_val,y_train,y_val = split_dataset(X,y)


# * We will split dataset because we will use lots of model so our training time can decrease for this dataset it's not very important actually because we dont't have a very big dataset. But we want to show it here to how to split and how can decide its correct splitted. We should divide dataset in a way that best expresses. 
# 
# 
# * For this we define function. The function  expects four arguments from us, these arguments: features,label, percent and frequency. Function can be split randomly by features and labels , after we will define for loop with number of frequency so we can select each parth of dataset. After we add a dict with error rate and loop number. We can show on graph each point. Graph will show us attempts on error rate. So we can show frequency. 

# In[418]:


def decide_split_number(X,y,percent,frequency,model = RandomForestRegressor(n_estimators=30)):
    """
    Params:
    X => dataset features
    y => dataset labels
    percent => percent of split
    frequency => the number of points to show in the graph
    model => model is our training model
    
    """
    n = int((len(X) * percent) / 100) 

    error_dict = {}
    
    for j in range(int(frequency)):
        
        idx = np.random.randint(0,1460,n)
        x_train = X[idx]
        y_train = y[idx]

        model.fit(x_train,y_train)

        y_array = np.mean(y_train) 
        y_avg = np.full((n,1),y_array) # average 
        y_pred = rnd.predict(x_train) # prediction

        # we will look at the error rate y_avg and y_pred
        # root mean squared error 
        total = 0
        for i in range(0,len(y_pred)):
            total += np.sqrt(np.square((y_avg - y_pred[i])))

        rmse = np.mean(total) / len(y_pred) 
        
        error_dict[j] = rmse

    y_axis = list(error_dict.values())
    
    x_axis = list(error_dict.keys())
    
    plt.plot(x_axis,y_axis,marker = '.',linewidth = 1.5,markersize = 8)
    plt.xlabel("Number Of Attempts")
    plt.ylabel("Error Rates")
    plt.title("The Frequency Of Errors To Try")
    plt.figure(figsize=(12,12))

    
    return plt.show()


# In[481]:


decide_split_number(X,y,20,30)


# # 7. Model Selection 

# In the next step, we will select our model, for this we will train the data set with many models. So we can choose one of the models with maximum success. For this we defined class this class has lots of method these methods: fit,predict,error_print. fit_method  purpose is training our model, predict method is predict our validation data. With the error print method, we will be able to calculate the error between our predicted values and our actual values and show them on the screen.

# In[421]:


class Model_Selection:
    def __init__(self,model,X,y):
        
        self.model = model
        
        self.x_train,self.x_val,self.y_train,self.y_val = split_dataset(X,y,rate = 0.2)
        
    def fit(self):
        return self.model.fit(self.x_train,self.y_train)
    
    def predict(self):
        return self.model.predict(self.x_val)
    
    def error_print(self,y_pred):
        
        print(f"Model Name: {str(self.model)[:-2]}\nR^2: \t{r2_score(self.y_val,y_pred)}")
        print("Mean Squared Error: {}".format(mean_squared_error(self.y_val,y_pred)))
        print("Root Mean Squared Error: {}".format(mean_squared_error(self.y_val,y_pred,squared=False)))
        print("Mean Absolute Error: {} ".format(mean_absolute_error(self.y_val,y_pred)))
        print("Mean Squared Log Error:{} ".format(mean_squared_log_error(self.y_val,y_pred)))
        print("*"*50)   


# In[422]:


# We define what we want to use as the models
model_list = [RandomForestRegressor(),DecisionTreeRegressor(),LinearRegression(),
             LGBMRegressor(),XGBRegressor(),CatBoostRegressor(verbose = 0),SVR()]

for model_name in model_list:
    
    model = Model_Selection(model_name,X,y)
    model.fit()
    y_pred  = model.predict()
    model.error_print(y_pred)


# * Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price.

# In[423]:


# Real and for our prediction we take  logarithm separately. 
# We will  the calculate RMSE between each other 
def log_mean_squared_err(y_pred,y_val):
    y_true_log = []
    y_pred_log = []
    for i in y_pred:
        y_pred_log.append(math.log(i))
    for i in y_val:
        y_true_log.append(math.log(i))
    return mean_squared_error(y_pred_log,y_true_log,squared=False)


# * Based on these results, we decided to choose XGBRegressor as the model. Next step we will  set the hyperparameters  for XGBRegressor.

# In[424]:


xgb = XGBRegressor(learning_rate=0.1,n_jobs=-1,n_estimators=500,max_depth=2,objective='reg:squarederror',
                   colsample_bytree=0.5,colsample_bylevel=0.9,colsample_bynode=0.75,min_child_weight=0.5)

xgb.fit(x_train,y_train)
y_pred = xgb.predict(x_val)

print(f"R^2:{r2_score(y_val,y_pred)}")
print("Mean Squared Error: {}".format(mean_squared_error(y_val,y_pred)))
print("Root Mean Squared Error: {}".format(mean_squared_error(y_val,y_pred,squared=False)))
print("Mean Absolute Error: {} ".format(mean_absolute_error(y_val,y_pred)))
print("Mean Squared Log Error: {} ".format(mean_squared_log_error(y_val,y_pred)))

# Calculating the values in rmse after taking their logarithm
log_rmse = log_mean_squared_err(y_pred,y_val)
print("RMSE Of Logarithmized Values: {}".format(log_rmse))


# # 8. Kfold Cross Validation

# In the next step we will train our dataset by applying k-fold cross validation. Thus we can train our model better.Because all dataset can be validation and training dataset. Thus we can evaluate our model more realistically. 

# In[452]:


# We will to split dataset to 4 because this way we can training each part of dataset.

kfold = KFold(n_splits=4,shuffle = True,random_state= 10)

log_RMSE  = []
fold_no = 0

for train_idx,val_idx in kfold.split(X):

    x_train,y_train = X[train_idx],y[train_idx]
    x_val,y_val = X[val_idx],y[val_idx]
    
    model = XGBRegressor(learning_rate=0.1,n_jobs=-1,n_estimators=500,max_depth=2,objective='reg:squarederror',
                   colsample_bytree=0.5,colsample_bylevel=0.9,colsample_bynode=0.75,min_child_weight=0.5)
    
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_val)
    
    log_RMSE.append(log_mean_squared_err(y_pred,y_val))
    
    fold_no += 1
    
    print("Fold No: {}".format(fold_no))
    print("R^2: {}".format(r2_score(y_val,y_pred)))
    print("Mean Squared Error: {}".format(mean_squared_error(y_val,y_pred)))
    print("Root Mean Squared Error: {}".format(mean_squared_error(y_val,y_pred,squared=False)))
    print("Mean Absoluate Error: {}".format(mean_absolute_error(y_pred,y_val)))
    print("Mean Squared Log Error: {}".format(mean_squared_log_error(y_pred,y_val)))


# # 9. Prepare Model For Submission

# Next step we will predict our test data. For the submission. But fisrtly we training train dataset with our XGBRegressor model. Because we previously trained our model by splitting , if we training whole train dataset our model make better a prediction.

# In[476]:


model.fit(X,y) # XGBREGRESSOR
y_pred = model.predict(X)
print("R^2: {}".format(r2_score(y,y_pred)))


# In[473]:


test_stack = model.predict(test_data.values)


# In[474]:


test_id = test_df["Id"]


# In[475]:


submission = pd.DataFrame({
        "Id": test_id,
        "SalePrice": test_stack})
submission.to_csv('submission.csv', index=False)

