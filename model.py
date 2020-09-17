import numpy as np
import pandas as pd


# loading the data
data = pd.read_csv('dataset.csv')


#imputing missing values for the categorical varaibles
cat_list = ['location_region', 'location_state','customer_value','gender','device_type','device_manufacturer']
for a in cat_list:
    data[a].fillna('unspecified',inplace=True)

    

#imputing missing values for the numerical variables
num_list = ['spend_total', 'spend_vas', 'spend_voice', 'spend_data','xtra_data_talk_rev', 'customer_class','age']
for b in num_list:
    data[b].fillna(data[b].median(), inplace=True)

data['sms_cost'].fillna(value=0, inplace=True)
data['event_type'].fillna(data['event_type'].mode()[0], inplace=True)
        

    
#Reducing the categories in device_manufacturer column        
lis = ['tecno','itel','infinix','samsung','nokia','apple']
for j in data['device_manufacturer']:
    if j not in lis:
        data['device_manufacturer'].replace(j,'others',inplace=True)
        

        
#Encoding the target variable
dict ={'Click': 1,'sms':0}
for i in dict:
    data['event_type'].replace(i,dict[i],inplace=True)

    
      
# Encoding the customer_value variable        
dict ={'unspecified': 0,'low':1,'medium': 2,'high':3,'very high': 4,'top':5}
for k in dict:
    data['customer_value'].replace(k,dict[k],inplace=True)
        


# import packages for encoding of the categorical variables - One Hot Encoding
X = pd.get_dummies(data, columns=cat_list, dummy_na=True)



#Dropping id's and empty variables
list = ['msisdn', 'location_lga','location_city', 'os_name','os_version',
       'ad_id', 'ad_name', '@timestamp', 'event_type']
y = X.loc[:,'event_type']
X = X.drop(list,axis = 1 )



# Model building using Random Forest algorithm
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=43)      
rf = rf.fit(X, y)

print('model ready')

import joblib
joblib.dump(rf, 'model.pkl')
print ('Model dumped!')


# Load the model that you just saved
rf = joblib.load('model.pkl')

model_columns = X.columns
joblib.dump(model_columns, 'model_columns.pkl')
print ('Model columns dumped!')