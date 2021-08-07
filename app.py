from flask import Flask, redirect, url_for, jsonify, request,render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def index():
   return render_template("index1.html")

@app.route('/predict',methods = ['GET','POST'])
def predict():
    values=[]
    stockname = request.form['name']
    openprice = float(request.form['open'])
    values.append(openprice)
    low = float(request.form['low'])
    values.append(low)
    high = float(request.form['high'])
    values.append(high)
    volume = int(request.form['volume'])
    values.append(volume)
    
    final_values = [np.array(values)]
    prediction = model.predict(final_values)
    result = prediction
    res = str(result[0])
    return render_template('index1.html',prediction_text=res)

if __name__ == '__main__':
   app.run(debug=True,use_reloader=False)
