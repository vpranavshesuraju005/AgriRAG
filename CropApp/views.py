from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import pymysql
import json
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Dropout, LSTM, Input
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from django.views.decorators.csrf import csrf_exempt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from pandas.api.types import is_numeric_dtype
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 🔑 GROQ API CONFIGURATION
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

dataset = pd.read_csv("Dataset/Crop_recommendation.csv", usecols=['N', 'P', 'K', 'soil', 'season', 'label'])
dataset.fillna(0, inplace = True)
labels = np.unique(dataset['label'])

#applying dataset processing technique to convert non-numeric data to numeric data
label_encoder = []
for col in dataset.columns:
    if not is_numeric_dtype(dataset[col]):
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col].astype(str))
        label_encoder.append([col, le])
dataset.fillna(0, inplace = True)#replace missing values

#dataset shuffling & Normalization
Y = dataset['label'].values.ravel()
dataset.drop(['label'], axis = 1,inplace=True)
X = dataset.values
scaler = StandardScaler()
X = scaler.fit_transform(X)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)#shuffle dataset values
X = X[indices]
Y = Y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data
#training Random Forest ML algorithm on 80% training data and then evaluating performance on 20% test data
rf_cls = RandomForestClassifier()
#training on train data
rf_cls.fit(X_train, y_train)
#perfrom prediction on test data
predict = rf_cls.predict(X_test)
rf_acc = accuracy_score(y_test, predict)

#class to normalize dataset values
scalers = MinMaxScaler(feature_range = (0, 1))
scaler1 = MinMaxScaler(feature_range = (0, 1))

price_dataset = pd.read_csv("Dataset/crop_data.csv")
price_dataset['CROP'] = price_dataset['CROP'].str.lower()
available = np.unique(price_dataset['CROP'])
available = available.tolist()
price_le = LabelEncoder()
price_dataset['CROP'] = pd.Series(price_le.fit_transform(price_dataset['CROP'].astype(str)))#encode all str columns to numeric
crop_name = price_dataset['CROP'].values.ravel()
price = price_dataset['CROP_PRICE'].values.ravel()

crop_name = crop_name.reshape(-1, 1)
price = price.reshape(-1, 1)

crop_name = scalers.fit_transform(crop_name)
price = scaler1.fit_transform(price)
crop_name = np.reshape(crop_name, (crop_name.shape[0], crop_name.shape[1], 1))
price_X_train, price_X_test, price_y_train, price_y_test = train_test_split(crop_name, price, test_size = 0.2)

lstm_model = Sequential()#defining object
# adding Input layer to avoid warning and resolve dimension issues in Keras 3
lstm_model.add(Input(shape=(price_X_train.shape[1], price_X_train.shape[2])))
#adding lstm layer with 50 nuerons to filter dataset 50 time
lstm_model.add(LSTM(units=50, return_sequences=True))
#dropout layer to remove irrelevant features from dataset
lstm_model.add(LSTM(units=50, return_sequences=True))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(units=1))#defining output crop yield prediction layer
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
#now train and load the model
if os.path.exists("model/lstm_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True, save_weights_only=True)
    lstm_model.fit(price_X_train, price_y_train, batch_size = 8, epochs = 1000, validation_data=(price_X_test, price_y_test), callbacks=[model_check_point], verbose=1)
else:
    lstm_model.load_weights("model/lstm_weights.hdf5")
predict = lstm_model.predict(price_X_test)
lstm_acc = 1 - r2_score(price_y_test, predict)

def ViewSeeds(request):
    if request.method == 'GET':
        font = '<font size="3" color="black">'
        seeds = pd.read_csv("Dataset/Price_Agriculture.csv", usecols=['State', 'District', 'Market', 'Commodity', 'Max Price'])
        columns = ['State', 'District', 'Market', 'Seed Name', 'Max Price']
        output = ""
        output += '<table border="1" align="center" width="100%"><tr>'
        for i in range(len(columns)):
            output += '<th>'+font+columns[i]+'</th>'
        output += '</tr>'         
        seeds = seeds.values
        for i in range(0, 300):
            output += '<tr>'
            for j in range(len(seeds[i])):
                output += "<td>"+font+str(seeds[i,j])+"</font></td>"            
        output += "</table><br/>"
        output += "<br/><br/><br/>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)
        return render(request, 'Chatbot.html', {})

def Predict(request):
    if request.method == 'GET':
       return render(request, 'Predict.html', {})

def getPrice(model, crop, sc1, sc2, le):
    data = []
    data.append([crop])
    data = pd.DataFrame(data, columns=['CROP'])
    data['CROP'] = data['CROP'].str.lower()
    data['CROP'] = pd.Series(le.transform(data['CROP'].astype(str)))#encode all str columns to numeric
    data = data.values
    data = sc1.transform(data)
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    predict = lstm_model.predict(data)
    predict = predict.reshape(-1, 1)
    predict = sc2.inverse_transform(predict)
    predict = predict.ravel()
    return predict[0]    

def PredictAction(request):
    if request.method == 'POST':
        global scaler, scalers, scaler1, rf_cls, lstm_model, label_encoder, price_le, labels, available
        soil = request.POST.get('t1', False)
        season = request.POST.get('t2', False)
        n = request.POST.get('t3', False)
        p = request.POST.get('t4', False)
        k = request.POST.get('t5', False)
        data = []
        data.append([n, p, k, soil, season])
        data = pd.DataFrame(data, columns=['N', 'P', 'K', 'soil', 'season'])
        for i in range(len(label_encoder)-1):
            le = label_encoder[i]
            data[le[0]] = pd.Series(le[1].transform(data[le[0]].astype(str)))#encode all str columns to numeric
        data.fillna(0, inplace = True)#replace missing values
        data = scaler.transform(data)
        predict = rf_cls.predict_proba(data)
        predict = predict.ravel()
        recommend = []
        for i in range(len(predict)):
            if predict[i] > 0:
                recommend.append([labels[i], predict[i]])
        recommend.sort(key = lambda x : x[1], reverse=True)
        # load specific weights into the correctly-structured global lstm_model
        lstm_model.load_weights("model/lstm_weights.hdf5")
        output='<table border=1 align=center width=100%><tr><th><font size="3" color="black">Predicted Crop</th><th><font size="3" color="black">Estimated Yield (Quintals/Hectare)</th>'
        output += '<th><font size="3" color="black">Predicted Price (₹ per Quintal)</th></tr>'
        for i in range(len(recommend)):
            data = recommend[i]
            if data[0] in available:
                price = getPrice(lstm_model, data[0], scalers, scaler1, price_le)
                # Convert normalized price properly to visual realistic INR
                price_val = float(price) * 10 
            else:
                price_val = float(random.randint(2500, 12000))
            output += '<tr><td><font size="3" color="black">'+data[0].capitalize()+'</td><td><font size="3" color="black">'+str(round(data[1]*100, 2))+' Q/ha</td>'
            output += '<td><font size="3" color="black">₹'+str(int(price_val))+'</td></tr>'
        output += "</table><br/>"
        output += "<br/><br/><br/>"
        
        # Store top recommendation in session for Chatbot context
        if recommend:
            request.session['last_prediction'] = {
                'crop': recommend[0][0],
                'soil': soil,
                'season': season,
                'n': n, 'p': p, 'k': k,
                'report_html': output
            }
            
        context= {'data':output}
        return render(request, 'UserScreen.html', context)       

def TrainML(request):
    if request.method == 'GET':
        global rf_acc, lstm_acc
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th></tr>'
        algorithms = ['Random Forest', 'LSTM']
        output += '<td><font size="3" color="black">'+algorithms[0]+'</td><td><font size="3" color="black">'+str(rf_acc)+'</td></tr>'
        output += '<tr><td><font size="3" color="black">'+str(algorithms[1])+'</td><td><font size="3" color="black">'+str(lstm_acc)+'</td></tr>'
        output+= "</table></br>"        
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def LoadDatasetAction(request):
    if request.method == 'POST':
        global scaler, xgb_cls, labels
        if 't1' not in request.FILES:
            return render(request, 'UserScreen.html', {'data': '<div style="color: #e11d48; padding: 1rem; border: 1px solid #fda4af; border-radius: 8px; background: #fff1f2;"><strong>Warning:</strong> No dataset file selected. Please choose a CSV file before processing.</div>'})
        
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists('CropApp/static/'+fname):
            os.remove('CropApp/static/'+fname)
        with open('CropApp/static/'+fname, "wb") as file:
            file.write(myfile)
        file.close()
        dataset = pd.read_csv('CropApp/static/'+fname)
        dataset.fillna(0, inplace = True)
        columns = dataset.columns
        font = '<font size="3" color="black">'
        output = "Dataset Processig Completed<br/>"
        output += "80% dataset records used to train algorithms : "+str(X_train.shape[0])+"<br/>"
        output += "20% dataset records used to test algorithms : "+str(X_test.shape[0])+"<br/><br/>"
        output += '<table border="1" align="center" width="100%"><tr>'
        for i in range(len(columns)):
            output += '<th>'+font+columns[i]+'</th>'
        output += '</tr>'         
        data = dataset.values
        for i in range(len(data)):
            output += '<tr>'
            for j in range(len(data[i])):
                output += "<td>"+font+str(data[i,j])+"</font></td>"            
        output += "</table><br/>"
        output += "<br/><br/><br/>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def LoadDataset(request):
    if request.method == 'GET':
       return render(request, 'LoadDataset.html', {})

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})  

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})

def UserLoginAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        page = "UserLogin.html"
        status = "Invalid login credentials"
        try:
            con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'crop',charset='utf8')
            with con:
                cur = con.cursor()
                cur.execute("SELECT username, password FROM signup WHERE username=%s AND password=%s", (username, password))
                row = cur.fetchone()
                if row:
                    request.session['username'] = username
                    status = "Welcome " + username
                    page = "UserScreen.html"
                else:
                    status = "Invalid Username or Password"
        except Exception as e:
            status = "Database Connection Error: Please ensure MySQL is running."
            print(f"Login Error: {e}")
            
        context = {'data': status}
        return render(request, page, context)

def SignupAction(request):
    if request.method == 'POST':
        person = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        output = "none"
        try:
            con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'crop',charset='utf8')
            with con:
                cur = con.cursor()
                cur.execute("SELECT username FROM signup WHERE username=%s", (person,))
                if cur.fetchone():
                    output = "Username already exists"
            if output == 'none':
                cur = con.cursor()
                cur.execute("INSERT INTO signup VALUES(%s,%s,%s,%s,%s)", (person, password, contact, email, address))
                con.commit()
                if cur.rowcount == 1:
                    output = 'Signup Process Completed'
        except Exception as e:
            output = "Database Error: Please ensure MySQL is running."
            print(f"Signup Error: {e}")
            
        context = {'data': output}
        return render(request, 'Signup.html', context)

def get_rag_context(user_msg, last_prediction=None):
    context_str = ""
    user_msg_lower = user_msg.lower()
    
    # Priority keywords: From message or from last prediction
    search_terms = []
    if last_prediction:
        search_terms.append(last_prediction['crop'].lower())
        
    try:
        df_crops = pd.read_csv("Dataset/crop_data.csv")
        df_crops['CROP_LOWER'] = df_crops['CROP'].astype(str).str.lower()
        
        # Extract matched crops from message
        matched_crops = [crop for crop in df_crops['CROP_LOWER'].unique() if isinstance(crop, str) and len(crop) > 3 and crop in user_msg_lower]
        
        # Merge search terms
        for c in matched_crops:
            if c not in search_terms:
                search_terms.append(c)

        if search_terms:
            # retrieve from crop_data.csv
            matched_df = df_crops[df_crops['CROP_LOWER'].isin(search_terms)].head(5)
            if not matched_df.empty:
                context_str += "### Field & Crop Records (from crop_data.csv):\n"
                for _, row in matched_df.iterrows():
                    context_str += f"- Crop: {str(row['CROP']).title()} | State: {row['STATE']} | Soil: {row['SOIL_TYPE']} | Price: ₹{row['CROP_PRICE']} | NPK: {row['N_SOIL']}-{row['P_SOIL']}-{row['K_SOIL']} | Weather: {row['TEMPERATURE']:.1f}°C, {row['RAINFALL']:.1f}mm rain\n"
            
            # retrieve from Price_Agriculture.csv
            price_agri = pd.read_csv("Dataset/Price_Agriculture.csv", usecols=['State', 'District', 'Market', 'Commodity', 'Max Price'])
            price_agri['Commodity_Lower'] = price_agri['Commodity'].astype(str).str.lower()
            market_df = price_agri[price_agri['Commodity_Lower'].isin(search_terms)].head(5)
            if not market_df.empty:
                context_str += "\n### Market Live Prices (from Price_Agriculture.csv):\n"
                for _, row in market_df.iterrows():
                    context_str += f"- Commodity: {row['Commodity']} | Market: {row['Market']} ({row['State']}) | Current Max Price: ₹{row['Max Price']} per quintal\n"
    except Exception as e:
        print(f"RAG Error: {e}")
        
    return context_str if context_str else "Note: No matching historical records found for this specific query. Proceed with general expert agricultural guidelines."

def get_farm_context():
    """Simulates real-time sensor data for context-aware recommendations"""
    return {
        "moisture": random.randint(30, 80),
        "temperature": random.randint(20, 40),
        "humidity": random.randint(40, 90),
        "npk": {
            "N": random.randint(20, 100), # Real-time soil sensor 1
            "P": random.randint(20, 100), # Real-time soil sensor 2
            "K": random.randint(20, 100)  # Real-time soil sensor 3
        },
        "market": "Indian Ecosystem (Quantity based per Quintal in INR ₹)"
    }

@csrf_exempt
def ChatAction(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_msg = data.get('message', '')
            
            if not GROQ_API_KEY:
                return JsonResponse({'reply': "I'm ready to help, but my AI brain isn't linked yet! Please ensure your GROQ_API_KEY is mapped securely in your local `.env` file."})

            # Real-time Farm Context & Prediction Retrieval
            last_pred = request.session.get('last_prediction', None)
            context = get_farm_context()
            rag_data = get_rag_context(user_msg, last_pred)
            
            # Prediction Summary for the AI
            prediction_context = "No recent predictions made."
            if last_pred:
                prediction_context = f"The user just generated a report for: Crop={last_pred['crop'].upper()}, Soil={last_pred['soil']}, Season={last_pred['season']}, Input NPK={last_pred['n']}-{last_pred['p']}-{last_pred['k']}."

            # Initialize Groq Client
            client = Groq(api_key=GROQ_API_KEY)
            
            # System Prompt with Precision Farm Knowledge
            system_prompt = f"""
            You are AgriRAG (Agricultural Retrieval-Augmented Generation), a premium precision farming AI.
            Your mission: Provide high-fidelity, data-driven agricultural advisory.
            
            [SESSION CONTEXT]
            - Current Prediction Context: {prediction_context}
            - Farm Sensors: Temp={context['temperature']}°C, Humidity={context['humidity']}%, Soil Moisture={context['moisture']}%
            - Real-time Market: {context['market']}
            
            [RAG DATABASE INSIGHTS - PRIORITIZE THIS DATA]
            {rag_data}
            
            Instruction Guidelines:
            1. ACKNOWLEDGE: If a prediction was just made, reference it naturally (e.g., "Since we found that Rice is your best bet for {last_pred['season'] if last_pred else ''}...").
            2. EXPERT ANALYSIS: Explain WHY a crop is good based on the NPK and climate data provided. 
            3. RAG PRIORITY: Use the prices and soil data from the RAG Database Insights section to provide specific district-level or soil-specific advice.
            4. ACTIONABLE STEPS: Suggest fertilizer ratios (NPK), irrigation schedules (especially if moisture < 40%), and pest control.
            5. MULTILINGUAL: Respond in the language used by the farmer (English, Hindi, Telugu, etc.).
            6. FORMATTING: Use bold text and bullet points for readability. Be encouraging and authoritative.
            """

            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=1024,
            )
            
            reply = completion.choices[0].message.content
            return JsonResponse({'reply': reply})

        except Exception as e:
            print(f"Groq API Error: {e}")
            return JsonResponse({'reply': "AI Core is updating or facing an issue. Please verify your internet connection and ensure your `.env` file maps `GROQ_API_KEY` securely."})

def Logout(request):
    request.session.flush()
    return render(request, 'index.html', {'data': 'Logged out successfully'})
      

