from flask import Flask, render_template, request
import pickle
import numpy as np

train = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.img['reader.result']
    data2 = request.img['reader1.result']
    # data3 = request.form['c']
    # data4 = request.form['d']
    arr = np.array([[data1, data2]])
    pred = train.predict(arr)
    return render_template('next.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)
