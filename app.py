from flask import Flask,render_template,redirect,url_for,request
import pickle
import numpy as np
import pandas as py
import sklearn
app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/help')
def help():
    return render_template("help.html")

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

@app.route("/disindex")
def disindex():
    return render_template("disindex.html")

@app.route("/gender")
def gender():
    return render_template("gender.html")

@app.route("/male_diabetes")
def male_diabetes():
    return render_template("male_diabetes.html")

@app.route("/female_diabetes")
def female_diabetes():
    return render_template("female_diabetes.html")

@app.route('/predict_male', methods=['POST'])
def predict_male():
    if request.method == 'POST':

        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']
        preg = 0

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        input_data_as_numpy_array=np.asarray(data)
        input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

        #loading the saved model
        loaded_model=pickle.load(open('C:/Users/sanje/OneDrive/Desktop/disease prediction/trained_model.sav','rb'))

        my_prediction=loaded_model.predict(input_data_reshaped)
        return render_template('diab_result.html', prediction=my_prediction)

@app.route('/predict_female', methods=['POST'])
def predict_female():
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']


        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        input_data_as_numpy_array=np.asarray(data)
        input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

        #loading the saved model
        loaded_model=pickle.load(open('C:/Users/sanje/OneDrive/Desktop/disease prediction/trained_model.sav','rb'))

        my_prediction=loaded_model.predict(input_data_reshaped)
        return render_template('diab_result.html', prediction=my_prediction)


@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route('/predictheart', methods=['POST'])
def predictheart():
        if request.method == 'POST':
            input_features = [float(x) for x in request.form.values()]
            features_value = [np.array(input_features)]
            features_name =[ "age", "sex", "cp", "trestbps","chol", "fbs", "restecg","thalach","exang","oldpeak","slope","ca","thal"]
            df = py.DataFrame(features_value, columns=features_name)
            loaded_model=pickle.load(open('C:/Users/hp/Desktop/Osmosys/flask/heart_disease_model.sav','rb'))
            my_prediction=loaded_model.predict(df)
            return render_template('heart_result.html', prediction=my_prediction)

@app.route("/parkinsons")
def parkinsons():
    return render_template("parkinsons.html")

@app.route('/predictparkinsons', methods=['POST'])
def predictparkinsons():
        if request.method == 'POST':
            input_features = [float(x) for x in request.form.values()]
            features_value = [np.array(input_features)]
            features_name =["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)",	"MDVP:Jitter(%)","MDVP:Jitter(Abs)",	"MDVP:RAP","MDVP:PPQ","	Jitter:DDP","	MDVP:Shimmer", "MDVP:Shimmer(dB)"," Shimmer:APQ3 "," Shimmer:APQ5 ","MDVP:APQ  ","Shimmer:DDA","	NHR","	HNR","	RPDE","	DFA","	spread1","	spread2","	D2","	PPE"] 
            df = py.DataFrame(features_value, columns=features_name)
            loaded_model=pickle.load(open('C:/Users/sanje/OneDrive/Desktop/disease prediction/parkinsons_model.sav','rb'))
            my_prediction=loaded_model.predict(df)
            return render_template('parkinsons_result.html', prediction=my_prediction)

@app.route('/terms')
def terms():
    return render_template("tc.html")


@app.route('/logout')
def logout():
    return redirect(url_for('index'))

if __name__== "__main__":
    app.run(debug=True)