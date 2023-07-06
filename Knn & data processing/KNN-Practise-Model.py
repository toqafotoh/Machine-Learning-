#!/usr/bin/env python
# coding: utf-8

# # Importing needed libraries 
# 
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns


# 
# 
# # loading data and discover it
# 

# In[2]:


data = pd.read_csv('survey.csv')
data.head(5)


# In[3]:


#saving columns names and number of them 
n_cols = data.shape[1]
col_names = data.columns
col_names


# ####  we see that column names form are inconsistent , there are nan values , and a lot of strings so that must be edited in cleaning data

# ## Let's take a look at the data 
# first we can figure out the spread of age values 

# In[4]:


plt.figure(figsize=(25,6))

# we will used value_count() to know the number of rows for each value on the dataset
age_counts = data['Age'].value_counts()

sns.barplot(x=age_counts.index, y=age_counts.values)
#sns.lineplot(data=data,y='treatment',x='Country')


# ### treatment will be our target so let's figure it out

# In[5]:


sns.barplot(y=data.index, x=data['treatment'])


#  okay, it's clear that most of people in this sample have sought a treatment for a mental health condition

# ### let's plot some columns to see thier values and percentage

# In[6]:


include_cols = ['self_employed',
       'family_history', 'treatment', 'work_interfere',
       'remote_work', 'tech_company', 'benefits', 'care_options',
       'wellness_program', 'seek_help']

fig, axs = plt.subplots(2, 5, figsize=(25, 8))

# Loop over each column in include_cols and plot it 
for i, col in enumerate(include_cols):
    row = i // 5
    col = i % 5
    '''as we create the figure size like a matrix with 2 rows and 5 columns
    we need to know the index of each figure by row and column so it will be updated each iteration'''
    
    # we will used value_count() to know the number of rows for each value on the dataset as we mentioned before
    counts = data[include_cols[i]].value_counts()

    
    axs[row, col].bar(counts.index, counts.values)
    axs[row, col].set_title(include_cols[i])

# making a space between plots
plt.subplots_adjust(hspace=0.5, wspace=0.2)

plt.show()


# # Cleaning Data
# as we see before , the data is not clean so we need to prepare data for analysis by removing or modifying data that is incorrect, incomplete, irrelevant, duplicated, or improperly formatted
# 
# this will be done within severel ways so let's starts 

# ## 1- drop unneeded columns
# 

# In[7]:


data = data.drop(['state','Country','Timestamp','comments'],axis=1)


# In[8]:


#saving the new column names and number
n_cols = data.shape[1]
col_names = data.columns
col_names


# In[9]:


n_cols


# ## 2- set the labels of the columns 
# 

# In[10]:


#for loop within all cols we saved before to make it all lowercase
for col in col_names :
    data.rename(columns={col: col.lower()}, inplace=True)
col_names = data.columns
col_names


# now no inconsistent column names 

# ## 3-  Handling Missing Values

# In[11]:


# get the number of missing data per each column
missing_count = data.isnull().sum()

# look at the number of missing points in the columns
missing_count[0:n_cols]


# In[12]:


#get the percentage of missing values
n_missing = missing_count.sum()
n_cells = np.product(data.shape)
(n_missing/n_cells)*100


#  ### filling missing values

# In[13]:


#at the self employed column there are small number of nan values and it someway equevlent to no so we will replace it
data['self_employed'] = data['self_employed'].fillna("No")
#the other nan values can be treated in many ways , i choose to drop them 
data = data.dropna()
data


# In[14]:


#calculating percentage of missing values after handling them
missing_count = data.isnull().sum()
n_missing = missing_count.sum()
n_cells = np.product(data.shape)
(n_missing/n_cells)*100


# No missing values now

# ## 4- Editing inconsistent data

# In[15]:


#get the unique values of column gender to check if there is inconsistent data
#data ['gender'].unique()
gender_v = pd.DataFrame(data ['gender'])
gender_v.value_counts()


# there are many weird values , we need it to be only female & male

# #### Setting all possible values to only Male & Female

# In[16]:


data.loc[data['gender'].str.contains('^M.*', regex=True, na=False), 'gender'] = 'Male'
data.loc[data['gender'].str.contains('Woman', regex=True, case=False, na=False), 'gender'] = 'Female'
data.loc[data['gender'].str.contains('F', case=False, na=False), 'gender'] = 'Female'
data.loc[data['gender'].str.contains('Man', regex=True, na=False), 'gender'] = 'Male'
data.loc[data['gender'].str.contains('^male', case=True,regex=True, na=False), 'gender'] = 'Male'
#data.loc[data['gender'].str.contains('Male', case=False,regex=True, na=False), 'gender'] = 'Male'


# In[17]:


data ['gender'].unique()


# the other values is unknown so we will drop it 

# In[18]:


data = data.drop(data[~data['gender'].isin(['Female', 'Male'])].index)
data ['gender'].unique()


# ### we need to check the inconsistency for each column

# In[19]:


for col in col_names :
    unique = data[col].unique()
    print(f"Unique values in column '{col}': {unique}")


# we see that the age column has negative values which dosen't make sense so we will apply a condition on it

# In[20]:


data = data[data['age'] >= 10]
data


# ## 5- Editing some datatypes
# no_employees column has a ranged values and it's type is string as we see

# In[21]:


data['no_employees'].unique()


# so we will split the string to take the minimum number for each row

# In[22]:


data_t = data
data.loc[data['no_employees'].str.contains('-', case=False, na=False), 'no_employees'] = data.loc[data['no_employees'].str.contains('-', case=False, na=False), 'no_employees'].str.split('-', expand=True)[0].astype(int)


# In[23]:


data['no_employees'].unique()


# In[24]:


#another way to do this
for i, row in data_t.iterrows():
    if '-' in str(row['no_employees']):
        data_t.at[i
                  , 'no_employees'] = int(str(row['no_employees']).split('-')[0])
data_t['no_employees'].unique()


# ### inconsistency still exist 
# so we will convert strings to the minimum too

# In[25]:


for i, row in data.iterrows():
    if 'More than 1000' in str(row['no_employees']):
        data.at[i, 'no_employees'] = 1000
#set column type to int
data['no_employees'] = data['no_employees'].astype(int)
data['no_employees'].unique()


# ## 6- encoding data
# encoding done by 2 ways : if you have columns which contains only 2 values like yes or no , true or false
# you can do binary encoding , 
# if there are more values so we will use one hot encoding 
# 
# as we see the data unique values before .. there is some columns which has only 2 values 
# so we will apply binary encoding at first 

# In[26]:


lb = LabelBinarizer()

for col in [ 'gender', 'self_employed', 'family_history', 'treatment','remote_work', 'tech_company','obs_consequence']:
    data[col] = lb.fit_transform(data[col])


# In[27]:


#now we need to split the data before one hot encoding to compare the error before & after 
not_hot = data.dtypes == int 
not_hot_cols = data.columns[not_hot]
not_hot_data = data.iloc[:, np.where(data.columns.isin(not_hot_cols))[0]]


# In[28]:


data.head(1).T


# now we see that some columns has been converted to int  

# In[29]:


#checking what will be the shape if we did one hot encoding
pd.get_dummies(data).shape


# we need to calculate number of don't know values to decide what to do 

# In[30]:


for col in col_names:
    num_dont_know = sum(data[col] == "Don't know")
    print(f"{col} has {num_dont_know} 'don't know' values")
    
## or data['col'].value_counts() will give us an over view for the data of the column 


# there are a lot of them , so we cannot drop it 

# In[31]:


# Select the string columns to do one hot encoding
strr = data.dtypes == object
s_cols = data.columns[strr]


# In[32]:


ex_cols = (data[s_cols]
                .apply(lambda x: x.nunique())
                .sort_values(ascending=False))
'''apply(lambda x: x.nunique()) applies the nunique() method to each column in the selected subset
and returns a pandas Series with the number of unique values in each column.'''
ex_cols -= 1
ex_cols.sum()
#number of additional cols 


# In[33]:


ex_cols
## to get the columns names u can try (ex_cols.index)
ex_cols.index


# In[34]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[35]:


data_copy = data.copy()


# In[36]:


#one hot encoding 
#first we willl loop over all columns that must be encoded  
for col in ex_cols.index:
    #first we will apply label encoder to assign unique number for each value
    col_x = LabelEncoder().fit_transform(data[col]).astype(int)
    #now we have to drop the column that will be replaced with some columns one for each value
    data_copy = data_copy.drop(col,axis=1)
    #now we apply one hot encoder on the selected column
    new_x = OneHotEncoder().fit_transform(col_x.reshape(-1,1))
    # we need to calculates the number of new columns
    n_new_x = new_x.shape[1]
    #givig each new column a nuique name
    name_new_x = ['_'.join([col, str(h)]) for h in range(n_new_x)]
    #create the new columns data frame
    new_df = pd.DataFrame(new_x.toarray(),index=data_copy.index,columns=name_new_x)
    #concatinate the new columns with the data 
    data_copy = pd.concat([data_copy,new_df],axis=1)


# In[37]:


#priint data after one hot encoding
data_copy


# In[38]:


##another way for one hot encoding

encoded_data = pd.get_dummies(data)
encoded_data
# this function get dummies apply one hot encoding to all of the categorical columns in the dataset

encoded_cols = list(encoded_data.columns)
data_2 = pd.concat([data, encoded_data], axis=1)
'''now we concatenated the encoded data with the original data 
so the original column and the encoded is exist too, we need to remove the orginal '''
data_2.drop(ex_cols.index, axis=1, inplace=True)

data_2


# In[39]:


data = data_copy
data


# In[40]:


data = data.astype(int)
data.dtypes


# # 6- scaling data

# In[41]:


# Create a figure with a single subplot
fig, ax = plt.subplots(figsize=(10, 7))

# Plot the histogram using ax
sns.histplot(data, bins=30, ax=ax)
ax.set_title("Original Data Distribution", fontsize=20)
ax.get_legend().remove()
plt.show()


# In[42]:


from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


# In[43]:


normalizer = Normalizer()
normalized_data = normalizer.fit_transform(data)


plt.figure(figsize=(10, 6))
sns.kdeplot(normalized_data[:, 0], shade=True)

plt.title("Normalized Data Distribution", fontsize=20)
plt.show()


# ### box-cox is a transformer used to change the distribution of the data to be normal ,
# it can work on positive data only so we needed to shift any non positive value to positive at first ,
# we apply shifting by subtract the minimum value from each value in the column ,
# then we add a small constant to each value so that no zeros in our dataset

# In[44]:


shifted_data = data - data.min(axis=0) + 1e-10

# Apply the Box-Cox transformation to the shifted data
transformer = PowerTransformer(method='box-cox', standardize=True)
transformed_data = transformer.fit_transform(shifted_data)

plt.figure(figsize=(10, 6))
sns.kdeplot(transformed_data[:, 0], shade=True)
plt.title("Box-Cox Transformed Data Distribution", fontsize=20)
plt.show()


# ##### StandardScaler
# is a technique for scaling data that transforms the data to have a mean of 0 and a standard deviation of 1
# by subtracting the mean of each feature and dividing by its standard deviation
# ##### MinMaxScaler
# is a technique for scaling data that transforms the data to a fixed range between 0 and 1
# by subtracting the minimum value of each feature and dividing by the range (maximum value minus minimum value) of each feature.
# ##### RobustScaler
# is a technique for scaling data that is more robust to outliers and non-normal distributions
# by subtracting the median of each feature and dividing by the interquartile range (IQR) of each feature.

# In[45]:


standard = StandardScaler()
standard_data = standard.fit_transform(data)

roubset = RobustScaler()
roubset_data = roubset.fit_transform(data)

minmax = MinMaxScaler()
minmax_data = minmax.fit_transform(data)

#sns.kdeplot(standard_data[:, 0], shade=True)
#plt.title("standard scalling Data Distribution", fontsize=20)
#plt.show()
fig, axs = plt.subplots(1, 3, figsize=(18, 7))

sns.kdeplot(data=standard_data[:, 0], ax=axs[0], shade=True)
sns.kdeplot(data=roubset_data[:, 0], ax=axs[1], shade=True)
sns.kdeplot(data=minmax_data[:, 0], ax=axs[2], shade=True)

axs[0].set_title('StandardScaled data', fontsize=16)
axs[1].set_title('RobustScaled data', fontsize=16)
axs[2].set_title('MinMaxScaled data', fontsize=16)

for i in [0,1,2] :
    axs[i].set_ylabel('Age')

plt.show()


# ##### convert these matrices to dataframes so we can work on it

# In[46]:


normalized_data = pd.DataFrame(normalized_data,columns=data.columns)
transformed_data = pd.DataFrame (transformed_data,columns=data.columns)
standard_data = pd.DataFrame(standard_data, columns=data.columns)
roubset_data = pd.DataFrame(roubset_data, columns=data.columns)
minmax_data = pd.DataFrame(minmax_data, columns=data.columns)


# # Spliting data 
# Now we need to split the data into train and test for each scaller

# In[47]:


diff_data = {'Normalized data' : normalized_data,
             'Transformed data' : transformed_data,
             'StandardScaled data' : standard_data,
             'RobustScaled data' : roubset_data,
             'MinMaxScaled data' : minmax_data,
             'Not hot-encoded data' : not_hot_data 
            }

y_col = 'treatment'

X_train_d = {}
X_test_d = {}
y_train_d = {}
y_test_d = {}

for key, value in diff_data.items():
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(value[[k for k in value.columns if k != y_col]],
                                                        data[y_col],
                                                        test_size=0.3,
                                                        random_state=42)
    # Store the training and testing sets into the corresponding dictionary
    X_train_d[key] = X_train
    X_test_d[key] = X_test
    y_train_d[key] = y_train
    y_test_d[key] = y_test


# # Applying KNN model

# In[48]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error


# In[49]:


errors = {}
#  a list to put errors in 
i=0
for n in [10,15,20,25]:
    knn = KNeighborsClassifier(n_neighbors=n, weights='distance',algorithm='kd_tree',p=1)
    for key, value in diff_data.items():
        knn = knn.fit(X_train_d[key], y_train_d[key])
        y_predict = knn.predict(X_test_d[key])
        cur_err = mean_squared_error(y_test_d[key],y_predict)
        if key in errors:
            errors[key].append(cur_err)
        else:
            errors[key] = [cur_err]

    errors = pd.Series(errors)
    print(f"errors for neighbors = {n}\n")
    for key, error_val in errors.items():
        print(key, error_val[i])
    print('-' * 80)
    i+=1


# In[50]:


g_list =list()

for name , arr in errors.items() :
    g_list.append((arr))
#g_list_transposed = list(zip(*g_list))

#for i in  [5,10,15,20,25]:
#    g_list_transposed.insert(0,i)
#g_list_transposed
g_list


# In[51]:


er_df = pd.DataFrame(g_list,columns=['10','15','20','25'])
er_df.index = diff_data.keys()
er_df_transposed = er_df.transpose()
er_df = pd.DataFrame(er_df_transposed)
er_df


# In[52]:


ax = er_df.plot(kind='line', legend=True, figsize=(15, 8))

ax.set_xlabel('n neighbors')
ax.set_ylabel('error',fontsize=16)
ax.set_title('The relation between n neighbors with errors')

plt.show()


# In[53]:


fig_er ={}
for n in range(1, 41):
    knn = KNeighborsClassifier(n_neighbors=n, weights='distance',algorithm='kd_tree',p=1)
    for key, value in diff_data.items():
        knn = knn.fit(X_train_d[key], y_train_d[key])
        y_predict = knn.predict(X_test_d[key])
        cur_err = mean_squared_error(y_test_d[key],y_predict)
        if key in fig_er:
            fig_er[key].append(cur_err)
        else:
            fig_er[key] = [cur_err]
    
    fig_er = pd.Series(fig_er)


# In[54]:


fig_er_list =list()
for name , arr in fig_er.items() :
    fig_er_list.append((arr))


# In[55]:


er_fig = pd.DataFrame(fig_er_list,columns=range(1, 41))
er_fig.index = diff_data.keys()
er_fig_transposed = er_fig.transpose()
er_fig = pd.DataFrame(er_fig_transposed)


# In[56]:


for col in er_fig.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(er_fig[col])
    plt.title(col)  
    plt.xlabel('n neighbors')
    plt.ylabel('error')
    plt.show()


# In[57]:


ax = er_fig.plot(kind='line', legend=True, figsize=(15, 8))

ax.set_xlabel('n neighbors')
ax.set_ylabel('error',fontsize=16)
ax.set_title('The relation between n neighbors with errors')

plt.show()

