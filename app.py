from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app=Flask(__name__)



curr=os.path.dirname(__file__)
model_path=os.path.join(curr, "Kidney_model_prediction.pkl")

with open(model_path,"rb") as f:
    classifier=pickle.load(f)



@app.route('/')

@app.route('/kidney')
def home():
    return render_template("kidney.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method=='POST':
        BloodPressure=float(request.form["BloodPressure"])
        Gravity=float(request.form["Gravity"])
        Albumin=float(request.form["Albumin"])
        BloodSugar=float(request.form["BloodSugar"])
        RedBlood=float(request.form["RedBlood"])
        PusCellCount=float(request.form["PusCellCount"])
        PusCellClump=float(request.form["PusCellClump"])


        input_data=(BloodPressure,Gravity, Albumin, BloodSugar,RedBlood, PusCellCount, PusCellClump)
        input_data_as_numpy_array=np.asarray(input_data)
        input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)
        prediction=classifier.predict(input_data_reshaped)

        if(prediction[0]==1):
            result="Sorry, you have chances of getting the disease. Please consult the doctor immediately."
        else:
            result="No need to fear. You have no dangerous symptoms of the disease."

        return render_template("result.html", result=result)
    

if __name__=='__main__':
    app.run(debug=True)
