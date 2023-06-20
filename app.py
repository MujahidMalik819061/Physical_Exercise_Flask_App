import joblib
from flask import Flask, render_templates, request
import pickle
import numpy as np
import pandas as pd
import gunicorn

filename = 'models.pkl'
classifier = pickle.load(open(filename,'rb'))
model = pickle.load(open('models.pkl','rb'))

app = Flask(_name_,templates_folder='Templates')

@app.route('/',methods=['GET'])
def index():
    return render_templates('index.html')

@app.route('/predict-value', methods=['POST'])
def predict_value():
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['jumping_jacks_down','jumping_jacks_up','pullups_down','pushups_up','squats_up','pullups_up','squats_down','situp_down','pushups_down','situp_up']

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    if output == 4:
        res_val = "Malign ! The person wrong way Physical_Exercise ! Please koi achhe trainer se training lo jo sahi tarike se sikhata ho."
    else:
        res_val = "Benign ! The person correct way to Physical_Exercise."
    return render_template('exerciseresult.html', prediction_text = 'person is{}'.format(res_val)) 

if _name_ == "_main_":
    app.run(debug=True)
