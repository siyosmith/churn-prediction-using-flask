from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/churn",methods=['GET','POST'])
def churn():
  
    tenure=request.form['tenure']
    Monthlycharge=request.form['Monthlycharge']
    arr=np.array([tenure,Monthlycharge])
    arr=arr.astype(np.float64)
    pred=model.predict([arr])

    if pred==1:
        result="yes"
    else:
        result="no"
    return render_template("index.html",prediction=result)








if __name__=='__main__':
    app.run(debug=True)