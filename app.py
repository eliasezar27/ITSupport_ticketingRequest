from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import numpy as np
import pandas as pd
import pickle


app = Flask(__name__)

@app.route('/', methods=["POST", "GET"])
def index1():
    return render_template('index.html')

@app.route('/result', methods=["POST", "GET"])
def index():
    feature_path = 'vect_feature.pkl'
    vect = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(feature_path, "rb")))
    cleanData = pd.read_pickle('cleanData.pkl')

    dt = joblib.load('finalized_model_decTree_Ctgry2.sav')
    dt2 = joblib.load('finalized_model_decTree_Assgn2.sav')

    # Input
    sampleString = request.form['desc']
    sampleString = ({sampleString})

    employeeId = float(request.form['emp_id'])
    vect_S_test = vect.transform(sampleString)

    # Decision Tree
    dt.predict_proba(vect_S_test)

    value = dt.predict_proba(vect_S_test)
    a = np.array(value)
    b = np.argmax(a) + 1
    e = a.max()
    e = e * 100

    category = cleanData.loc[cleanData['Category_Num'] == b]['Category_Level3Desc'].values
    category = category[0]

    # Assignee Classification

    # Decision Tree
    ar = [[b, employeeId]]
    value2 = dt2.predict_proba(ar)
    c=np.array(value2)
    d=np.argmax(c)
    f=c.max()
    f=f*100
    assignee = cleanData.loc[cleanData['Assign_Num'] == d]['AssigneeName'].values
    assignee = assignee[0]

    return render_template('index.html',  category = category, assignee = assignee)

if __name__ == '__main__':
    app.run(debug=True)