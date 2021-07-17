from flask import Flask, redirect, url_for, jsonify, request,render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
   return render_template("index.html")

@app.route('/ff',methods = ['POST', 'GET'])
def ff():
   return render_template("predict.html")

@app.route('/predict',methods = ['POST'])
def predict():
   values=[]  
   stockname=request.form['name'] 
   openprice = request.form['open']
   values.append(openprice)
   low = request.form['low']
   values.append(low)
   high = request.form['high']
   values.append(high)
   volume = request.form['volume']
   values.append(volume)
    
   final_values = [np.array(values)]
   prediction = model.predict(final_values)
   result = prediction
   res = str(result[0])
   return {'message':"stock prediction is"+res}
if __name__ == '__main__':
   app.run(debug=True,use_reloader=False)
