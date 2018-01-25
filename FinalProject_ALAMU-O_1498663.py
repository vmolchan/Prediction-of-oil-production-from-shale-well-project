
# coding: utf-8

# In[144]:

# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

from mpl_toolkits.mplot3d import Axes3D # Creating 3D plots
from sklearn.feature_selection import RFE # Recursive feature extraction
from sklearn.linear_model import LinearRegression # Linear regression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler # Scaling data
from sklearn import metrics # Calculating accuracy metrics
from sklearn.svm import SVR # Support vector regressor
from sklearn.tree import DecisionTreeRegressor # Decision tree regressor

from sklearn.neural_network import MLPRegressor # Multilayer perceptron


# In[145]:

# Import the training dataset
data = pd.read_excel('IntroEngDataScienceFinalProjectTrainingData.xlsx')


# In[146]:

print(data.head())
print('--------------------------------------------------------------')
print(data.isnull().sum()) # Check for missing values


# Basic Statistics and data cleaning.

# In[147]:

# show statistics
print(data.describe())


# In[148]:

# drop rows with missing values
data.dropna(inplace = True,axis = 0 )
# Check for missing values
print(data.isnull().sum())


# In[149]:

print('-----------------------------------------------------------------------------------')
# print a list of column names
print(data.columns.tolist())


# In[150]:

# Remove all text columns from the dataset
data=data.drop(labels=['JOB_DESC_STAGING','PROPPANT_MESH_DESCRIPTION','PROPPANT_MASS_UOM',
          'AVERAGE_STP_UOM','FRACTURE_GRADIENT_UOM','MD_MIDDLE_PERFORATION_UOM','MIN_STP_UOM',
          'MAX_STP_UOM'], axis = 1)


# Some visualization before aggregation

# In[151]:

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =data['WELL_LATITUDE']
y =data['WELL_LONGITUDE']
z =data['LIQ_CUM_BBLS']

ax.scatter(x, y, z, c=data['WELL_ID'], marker='o')

ax.set_xlabel('WELL_LATITUDE')
ax.set_ylabel('WELL_LONGITUDE')
ax.set_zlabel('LIQ_CUM_BBLS')
#plt.savefig(fname = '3Dimage')
plt.show()


# Aggregate the data by the well ID using the 'groupby' function in pandas

# In[152]:

# group the data by the well ID
Grouped_data =data.groupby('WELL_ID', as_index = False)
# sums all the properties by well ID
Grouped_data=Grouped_data.sum()
print(Grouped_data.head())
print('-----------------------------------------------------------------------------------')


# In[153]:

print(Grouped_data.info())


# The entire dataset has been grouped into 20 wells based on the well IDS present. This aggregation was done by summing all the columns per well id. This process is valid for quantities such as Volume of proppant, mass of proppant etc but is meaningless for quantities such as pressure, longitude, fracture gradient etc. In order to solve this, the mean values would be calculated for some other columns instead while the sum would be used for others on a case by case basis.

# In[154]:

# The number of rows per well id needs to be computed, 
#this value would then be used to compute the mean value.


# The function below was created by Olabode Alamu to count the number of rows that have info for a well id.

# In[155]:

"""
This function was created to count the number of rows in the dataset 
which partains to a particular well ID number.
it takes in the dataframe of interest as input, counts the number of rows per well id 
and returns a dataframe with the number of rows per well ID as output
"""
def rowcount(dataframe):
    Unique = dataframe['WELL_ID'].unique() # Checks for the unique well IDs
    length_list = [] 
    # slices through the dataframe till only unique Well ids are found and counted
    for i in Unique:
        length=len(dataframe[dataframe['WELL_ID']== i]) 
        length_list.append(length) # appends the count to the list
    
    # pass into a dataframe
    Count = pd.DataFrame(data= length_list, columns = ['No of rows'])
    Count['Well ID']= Unique
    return Count
   


# In[156]:

# the dataframe was passed into the function
Count=rowcount(data)


# In[157]:

print(Count)


# Next we compute the mean value for some of the columns

# In[158]:

# Create a column in the Grouped data with the number of rows
Grouped_data['Count'] = Count['No of rows']


# In[159]:

print(Grouped_data.head())


# The count column would then be used to compute the mean values for ['GAS_CUM','LIQ_CUM_BBLS','NET_PROD_DAYS','WELL_HORZ_LENGTH',
# 'TRUE_VERTICAL_DEPTH','LOWER_PERF','UPPER_PERF','MAX_STP',
# 'MIN_STP','WELL_LONGITUDE','WELL_LATITUDE',
#  'TOP_DEPTH','TVD_DEPTH','MD_MIDDLE_PERFORATION',
#  'FRACTURE_GRADIENT','AVERAGE_STP']

# In[160]:

# This would be achieved using a custom function


# In[161]:

def mean_calculator(dataframe,new_column_names,old_column_names):
    """
    This function takes in a dataframe and creates new columns based 
    on the calculated mean of the previous columns. it requires a list of the new column names and
    a list of the old column names which require a mean to be computed.
    """
    
    for i,j in zip(new_column_names,old_column_names):
        dataframe[i]= dataframe[j]/dataframe['Count']


# In[162]:

# list of columns which the mean to be computed
old_column_names = ['GAS_CUM','LIQ_CUM_BBLS','NET_PROD_DAYS','WELL_HORZ_LENGTH','TRUE_VERTICAL_DEPTH'
                   ,'LOWER_PERF','UPPER_PERF','MAX_STP','MIN_STP','WELL_LONGITUDE','WELL_LATITUDE',
                   'TOP_DEPTH','TVD_DEPTH','MD_MIDDLE_PERFORATION','FRACTURE_GRADIENT','AVERAGE_STP']


# In[163]:

# new column names that would be created on the dataframe
new_column_names=['Mean Gas_cum','Mean Liquid produced','Mean Production days','Mean Horizontal length',
 'Mean True Vertical Distance','Mean Lower perforation','Mean Upper perforation','Mean Maximum STP',
'Mean Minimum STP','Longitude','Latitude','Mean TOP Depth','Mean TVD depth','Mean Mid perforation',
 'Mean Fracture Gradient','Mean STP']


# In[164]:

# call the mean calculator custom function
mean_calculator(dataframe=Grouped_data,
                new_column_names=new_column_names,old_column_names=old_column_names)


# In[165]:

print(Grouped_data.head())


# In[166]:

# Create list of columns that were the sum of properties, remove them and replace with the mean property
remove_columns = ['AVERAGE_STP','FRACTURE_GRADIENT','MD_MIDDLE_PERFORATION','TVD_DEPTH',
                 'TOP_DEPTH','WELL_LATITUDE','WELL_LONGITUDE','MIN_STP','MAX_STP','UPPER_PERF',
                 'LOWER_PERF','TRUE_VERTICAL_DEPTH','WELL_HORZ_LENGTH','NET_PROD_DAYS','LIQ_CUM_BBLS'
                 ,'GAS_CUM']

# Drop some more columns
Grouped_data=Grouped_data.drop(labels=remove_columns, axis = 1)
    


# In[167]:

print(Grouped_data.head())


# In[168]:

plt.scatter(x= Grouped_data['VOLUME_PUMPED_GALLONS'], y = Grouped_data['Mean Liquid produced'],c='b')
plt.ylabel('Liquid produced in BBLs')
plt.xlabel('Volume of proppant in gallons')
plt.grid()
plt.show()


# In[169]:

plt.scatter(x= Grouped_data['VOLUME_PUMPED_GALLONS'], y = Grouped_data['Mean Gas_cum'])
plt.ylabel('Gas produced in BBLs')
plt.xlabel('Volume of proppant in gallons')
plt.grid()
plt.show()


# In[170]:

plt.scatter(x= Grouped_data['PROPPANT_MASS_USED'], y = Grouped_data['Mean Liquid produced'], c='y')
plt.ylabel('Liquid produced in BBLs')
plt.xlabel('Mass of proppant in CWT')
plt.grid()
plt.show()


# In[171]:

plt.scatter(x= Grouped_data['Mean TVD depth'], y = Grouped_data['Mean Liquid produced'], c = 'r')
plt.ylabel('Liquid produced in BBLs')
plt.xlabel('True vertical depth in feet')
plt.grid()
plt.show()


# In[172]:

plt.scatter(x= Grouped_data['Mean Upper perforation'], y = Grouped_data['Mean Liquid produced'])
plt.ylabel('Liquid produced in BBLs')
plt.xlabel('Upper perforation length in feet')
plt.grid()
plt.show()


# In[173]:

plt.scatter(x= Grouped_data['Mean Maximum STP'], y = Grouped_data['Mean Liquid produced'])
plt.ylabel('Liquid produced in BBLs')
plt.xlabel('Maximum pressure psi')
plt.grid()
plt.show()


# In[174]:

plt.scatter(x= Grouped_data['Mean STP'], y = Grouped_data['Mean Liquid produced'])
plt.ylabel('Liquid produced in BBLs')
plt.xlabel('Average pressure of fracking psi')
plt.grid()
plt.show()


# In[175]:

plt.scatter(x= Grouped_data['Mean Gas_cum'], y = Grouped_data['Mean Liquid produced'])
plt.ylabel('Liquid produced in BBLs')
plt.xlabel('Gas produced')
plt.grid()
plt.show()


# In[176]:

plt.scatter(x= Grouped_data['Mean Horizontal length'], y = Grouped_data['Mean Liquid produced'])
plt.ylabel('Liquid produced in BBLs')
plt.xlabel('Well horizontal length')
plt.grid()
plt.show()


# In[ ]:




# The dataframe is in the required format, we can continue with the analysis.

# The exported lat and long data would be used to create the basin location map

# In[177]:

#Export the longitude and latitude data
long_lat =Grouped_data[['WELL_ID','Latitude','Longitude', 
                        'Mean Liquid produced','Mean Production days']]
# export to a csv
long_lat.to_csv('long_lat.csv')


# In[178]:

# Create target variable
y = Grouped_data['Mean Liquid produced']

# These features are dropped because they are repetitve
dropoff = ['Mean Gas_cum','Count','WELL_ID','Latitude','Longitude',
           'Mean Liquid produced','Mean True Vertical Distance']

# Create input features
X=Grouped_data.drop(labels=dropoff, axis = 1)


# Feature selection was perfromed using the recursive feature extraction model in scikit learn

# The recursive feature extraction model requires an estimator, in this case, the linear regression model was chosen as the estimator since the objective of this project is that of a regression problem.

# In[179]:

lm = LinearRegression()
# SPlit the data into training and test data

X_train_rfe, X_test_rfe, y_train_rfe, y_test_rfe = train_test_split(X,y, test_size = 0.3)
# create the RFE model and select 12 attributes out of 13 possible
rfe = RFE(estimator = lm, n_features_to_select=12, verbose=3)
rfe = rfe.fit(X_train_rfe, y_train_rfe)
# summarize the selection of the attributes
print(rfe.support_) # the parameters with True are selected
print('----------------------------------------------------------------')
print(rfe.ranking_)


# Display the selected features

# In[180]:

# display important features
print(X.columns[rfe.support_])


# The selected features are ['PROPPANT_MASS_USED', 'Mean Production days', 'Mean Horizontal length',
#        'Mean Lower perforation', 'Mean Upper perforation', 'Mean Maximum STP',
#        'Mean Minimum STP', 'Mean TOP Depth', 'Mean TVD depth',
#        'Mean Mid perforation', 'Mean Fracture Gradient', 'Mean STP']

# The 'VOLUME_PUMPED_GALLONS' was dropped since it is the least important feature acording to the analysis.

# In[181]:

# The top 12 important features were selected
# They would be used to form the new X matrix
X = X[['PROPPANT_MASS_USED', 'Mean STP', 'Mean Fracture Gradient',
       'Mean Mid perforation', 'Mean TVD depth', 'Mean TOP Depth',
       'Mean Minimum STP', 'Mean Maximum STP', 'Mean Upper perforation',
       'Mean Lower perforation', 'Mean Horizontal length',
       'Mean Production days']]


# Prepare the X matrix by Standardizing it.

# In[182]:

# Standardize the data

Scaled = StandardScaler()
Scaled.fit(X)
Scaled.transform(X)


# In[183]:

# SPlit the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)


# Regression analysis

# In[184]:

lm = LinearRegression() # linear regression

DTR = DecisionTreeRegressor(max_depth=1) # Decision tree regressor
MLPR = MLPRegressor(max_iter = 200, solver = 'lbfgs', verbose=True, tol = 0.000001) # Multilayer perceptron


# In[185]:

SVR = SVR(C = 0.0001, epsilon = 0.2,kernel = 'linear') # Support vector regression


# In[186]:

def Regression_analysis(Regressor,X_train,y_train,X_test,y_test):
    Regressor.fit(X_train,y_train)
    Predict = Regressor.predict(X_test)
    plt.scatter(y_test,Predict)
    plt.xlabel('Y test values')
    plt.ylabel('Predicted values')
    plt.grid()
    plt.show()
    print('MAE:', metrics.mean_absolute_error(y_test, Predict))
    print('MSE:', metrics.mean_squared_error(y_test, Predict))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, Predict)))


# Linear regression

# In[187]:

Regression_analysis(lm,X_train,y_train,X_test,y_test)


# Support Vector Regression

# In[188]:

Regression_analysis(SVR,X_train,y_train,X_test,y_test)


# In[189]:

Regression_analysis(DTR,X_train,y_train,X_test,y_test)


# In[190]:

Regression_analysis(MLPR,X_train,y_train,X_test,y_test)


# In[ ]:




# In[ ]:




# Import test model datasheet

# In[125]:

test_data = pd.read_excel('IntroEngDataScienceFinalProjectTestModelOutput.xlsx')
print(test_data.head())
print(test_data.isnull().sum())


# In[126]:


test_data=test_data.drop(labels=['JOB_DESC_STAGING','PROPPANT_MESH_DESCRIPTION','PROPPANT_MASS_UOM',
          'AVERAGE_STP_UOM','FRACTURE_GRADIENT_UOM','MD_MIDDLE_PERFORATION_UOM','MIN_STP_UOM',
          'MAX_STP_UOM'], axis = 1)


# In[127]:

# Count the number of rows
Count = rowcount(test_data)


# In[128]:

print(Count)


# In[129]:

# group the data by the well ID
test_group =test_data.groupby('WELL_ID', as_index = False)
test_group=test_group.sum()
test_group['Count'] = Count['No of rows']
print(test_group.head())


# In[130]:

old_column_names = ['NET_PROD_DAYS','WELL_HORZ_LENGTH','TRUE_VERTICAL_DEPTH'
                   ,'LOWER_PERF','UPPER_PERF','MAX_STP','MIN_STP','WELL_LONGITUDE','WELL_LATITUDE',
                   'TOP_DEPTH','TVD_DEPTH','MD_MIDDLE_PERFORATION','FRACTURE_GRADIENT','AVERAGE_STP']
# new column names
new_column_names=['Mean Production days','Mean Horizontal length',
 'Mean True Vertical Distance','Mean Lower perforation','Mean Upper perforation','Mean Maximum STP',
'Mean Minimum STP','Longitude','Latitude','Mean TOP Depth','Mean TVD depth','Mean Mid perforation',
 'Mean Fracture Gradient','Mean STP']


# In[131]:

# Compute the mean and generate new columns using the predefined columns
mean_calculator(test_group,old_column_names=old_column_names,new_column_names=new_column_names)


# In[132]:

# Drop some  columns
test_group=test_group.drop(labels= ['AVERAGE_STP','FRACTURE_GRADIENT','MD_MIDDLE_PERFORATION','TVD_DEPTH',
                 'TOP_DEPTH','WELL_LATITUDE','WELL_LONGITUDE','MIN_STP','MAX_STP','UPPER_PERF',
                 'LOWER_PERF','TRUE_VERTICAL_DEPTH','WELL_HORZ_LENGTH','NET_PROD_DAYS'
                 ], axis = 1)


# In[133]:

Xtest=test_group.drop(labels=['Count','WELL_ID','Latitude','Longitude','Mean True Vertical Distance'], axis = 1)


# In[134]:

# re-index the column based on the selected features
Xtest = Xtest[['PROPPANT_MASS_USED', 'Mean STP', 'Mean Fracture Gradient',
       'Mean Mid perforation', 'Mean TVD depth', 'Mean TOP Depth',
       'Mean Minimum STP', 'Mean Maximum STP', 'Mean Upper perforation',
       'Mean Lower perforation', 'Mean Horizontal length',
       'Mean Production days','LIQ_CUM_BBLS']]


# In[135]:

print(Xtest)


# In[136]:

# Remove the liq cum column prior to Standardizing
Xtestscale = Xtest[['PROPPANT_MASS_USED', 'Mean STP', 'Mean Fracture Gradient',
       'Mean Mid perforation', 'Mean TVD depth', 'Mean TOP Depth',
       'Mean Minimum STP', 'Mean Maximum STP', 'Mean Upper perforation',
       'Mean Lower perforation', 'Mean Horizontal length',
       'Mean Production days']]


# In[137]:

#Standardize
Scaled.fit(Xtestscale)
Scaled.transform(Xtestscale)


# We use the multilayer perceptron to predict the liquid produced column 

# In[138]:

Xtest['LIQ_CUM_BBLS'] = MLPR.predict(Xtestscale)


# In[139]:

print(Xtest)


# In[140]:

Xtest['Well ID'] = test_group['WELL_ID']


# In[141]:

print(Xtest)


# In[142]:

Xtest


# Export the result to a csv file

# In[143]:

# export to a csv
Xtest.to_csv('FinalResult.csv')


# In[ ]:




# In[ ]:




# In[ ]:



