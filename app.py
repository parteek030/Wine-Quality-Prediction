import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
wine_regression = pickle.load(open('wine_regression.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('test_page.html')

@app.route('/predict',methods=['POST'])
def predict():

    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = wine_regression.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('test_page.html', prediction_text='Wine quality should be approximately $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)