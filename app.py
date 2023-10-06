from flask import Flask,request,jsonify,render_template 
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

#import ridg regresssor an standard scaler
ridge_model=pickle.load(open('MODELS/ridge.pkl','rb'))
Standard_Scaler=pickle.load(open('MODELS/scaler.pkl','rb'))

#route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature = float(request.form.get('Temprature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('WS'))
        Rain = float(request.form.get('RAIN'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('CLASSES'))
        Region = float(request.form.get('REGION'))

        new_data_scaled = Standard_Scaler.transform([[Temperature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]])

        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')





if __name__=="__main__":
    app.run(host="0.0.0.0")
