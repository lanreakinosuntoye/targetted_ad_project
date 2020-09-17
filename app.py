# Dependencies
from flask import Flask, request, jsonify
import json
import joblib
import traceback
import pandas as pd
import numpy as np



rf = joblib.load("model.pkl") # Load "model.pkl"
print ('Terragon Model loaded')
model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
print ('Terragon Model columns loaded')



# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            #content_type='application/json'
            #
            #jsdata = open('application/json', 'rb')
            #
            #jsondata = request.json
            #jsdata = pd.read_json(json.dumps(jsdata),orient='index')
          
            
            
            
            
            data = request.files['file']
            #data = open(data.filename, 'rb')
            #ata =  json.dumps(data)
            #data = json.loads(data)
            
            
            with open(data.filename, 'rb') as f:
                data = f.read()
            
            #data = pd.read_csv('file')
            data = pd.DataFrame(eval(data))      
         
             # Imputing null values
            cat_list = ['location_region','location_state', 'customer_value','gender','device_type','device_manufacturer']
            
            cust_value_list = ['low','medium','high','very high','top', ]
                    
            gender_list = ['M', 'F', ]
                    
            region_list = ['South West','South South','North Central','North West','South East','North East']
                    
            state_list =['lagos','ogun','rivers','fct','kano','kaduna','delta','oyo',
                         'kogi','abia','imo','anambra','edo','osun','enugu','ondo','bauchi','niger','katsina',
                         'akwa ibom','ekiti','borno','plateau','nassarawa','kwara','benue','cross river',
                         'sokoto','adamawa','gombe','yobe','taraba','kebbi','zamfara','ebonyi','bayelsa','jigawa']
            
            # Imputing null values for the categorical variable
            for a in cat_list:
                data[a].fillna('unspecified',inplace=True)
                
                #TypeError handling for all categorical variables
                for d in data[a]:
                    if isinstance(d, float) or isinstance(d, int):
                        raise TypeError({'value must be a string': traceback.format_exc()})
                    
                    #ValueError handling for the locfation_region incoming data
                    if a == 'location_region' and d not in region_list:
                        raise ValueError({'Unidentified Region'})                    

                    #ValueError handling for the locfation_region incoming data
                    if a == 'location_state' and d not in state_list:
                        raise ValueError({'Unidentified State'})                    

                     
                    #ValueError handling for the customer_value incoming data
                    if a == 'customer_value' and d not in cust_value_list:
                        raise ValueError({'Specify the right customer value'})
                    
                    #ValueError handling for the gender incoming data
                    if a == 'gender' and d not in gender_list:
                        raise ValueError({'Gender can only be M or F'})
                    
                    
            # Numerical variable variable list        
            num_list = ['spend_total', 'spend_vas', 'spend_voice', 'spend_data','sms_cost','xtra_data_talk_rev', 'customer_class','age']
          
            # Imputing null values for the numerical variable
            for b in num_list:
                data[b].fillna(value=0, inplace=True)
                
                #Error handling for numerical variables
                for f in data[b]:
                    if isinstance(f, str):
                        raise TypeError({'value must be an integer or float'})
                    
                    if b == 'age' and isinstance(f, float):
                        raise TypeError({'Age must be discrete, not continous'})
                    
                    if b == 'age' and f < 0:
                            raise ValueError({'Age cannot be negative'})
                    
                    

            #Reducing the categories in device_manufacturer column        
            lis = ['tecno','itel','infinix','samsung','nokia','apple']
            for j in data['device_manufacturer']:
                if j not in lis:
                    data['device_manufacturer'].replace(j,'others',inplace=True)


            # Encoding the customer_value variable        
            dict ={'unspecified': 0,'low':1,'medium': 2,'high':3,'very high': 4,'top':5}
            for k in dict:
                data['customer_value'].replace(k,dict[k],inplace=True)
     
                
             
            
            data = pd.get_dummies(data, columns=cat_list, dummy_na=True)
            data = data.reindex(columns=model_columns, fill_value=0)

            prediction = list(rf.predict(data))

            return jsonify({'prediction': str(prediction)})


       


        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
    

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 8080 # If you don't provide any port the port will be set to 

    rf = joblib.load("model.pkl") # Load "model.pkl"
    print ('Terragon Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Terragon Model columns loaded')

    app.run(port=port, debug=True)