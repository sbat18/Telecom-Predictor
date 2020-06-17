import numpy as np
from flask import Flask, request, render_template
import pickle
import keras
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = pickle.load(open('model1.pickle', 'rb'))
sc=pickle.load(open('model2.pickle','rb'))
global graph
graph = tf.get_default_graph() 

@app.route('/')
def home():
    return render_template('checkbinary.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
   
    
    val1=request.form['SeniorCitizen']
    val1=float(val1)
    a=np.array(val1)
    
    val2=request.form['Partner']
    val2=float(val2)
    a=np.append(a,val2)
    
    val3=request.form['Dependents']
    val3=float(val3)
    a=np.append(a,val3)
    
    val4=request.form['tenure']
    val4=float(val4)
    a=np.append(a,val4)
    
    val5=request.form['InternetService']
    val5=float(val5)
    a=np.append(a,val5)
    
    val6=request.form['OnlineSecurity']
    val6=float(val6)
    a=np.append(a,val6)
    
    val7=request.form['onlinebackup']
    val7=float(val7)
    a=np.append(a,val7)
    
    val8=request.form['DeviceProtection']
    val8=float(val8)
    a=np.append(a,val8)
    
    val9=request.form['TechSupport']
    val9=float(val9)
    a=np.append(a,val9)
    
    val10=request.form['Contract']
    val10=float(val10)
    a=np.append(a,val10)
    
    val11=request.form['PaperlessBilling']
    val11=float(val11)
    a=np.append(a,val11)
    
    val12=request.form['PaymentMethod']
    val12=float(val12)
    a=np.append(a,val12)
    
    val13=request.form['MonthlyCharges']
    val13=float(val13)
    a=np.append(a,val13)
    
    val14=request.form['TotalCharges']
    val14=float(val2)
    a=np.append(a,val14)
    

    
    a=a.reshape(1,14)
    a=sc.transform(a)
    with graph.as_default(): 
        prediction=model.predict(a)
        prediction=float(prediction)
        prediction=np.round(prediction,decimals=3)
        return render_template('checkbinary.html', prediction_text='Probability of this customer churn is {:%}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
    
    