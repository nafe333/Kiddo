from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import librosa
from keras.models import Model, Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

app = Flask(__name__)

sample_r = 56000
weights_path = 'Model.py'
classes = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
#######################################################################
@app.route('/')
def home():
    return  render_template('index.html',appName="HD prediction")
###########################################################################
def create_model():
    model = Sequential()
    model.add(Conv1D(16, 13, padding='valid', activation='relu', input_shape=(sample_r, 1)))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.3))
    model.add(Conv1D(32, 11, padding='valid', activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.3))
    model.add(Conv1D(64, 9, padding='valid', activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.3))
    model.add(Conv1D(128, 7, padding='valid', activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(6, activation='softmax'))
    return model

model = create_model()
##model.load_weights(weights_path)

##########################################################################

def preprocess_wav_file(wav_path, sample_r=8000, target_length=56000):
    print("preprocess_wav_file is work")
    samples, sample_rate = librosa.load(wav_path, sr=sample_r, mono=True)
    if len(samples) < target_length:
        samples = np.pad(samples, (0, target_length - len(samples)), 'constant')
    elif len(samples) > target_length:
        samples = samples[:target_length]

    return samples.reshape(1, -1, 1)

############################################

def predictmodel(audio):
    print("predict is work ")
    prob = model.predict(audio.reshape(1, sample_r, 1))
    index = np.argmax(prob[0])
    return classes[index]

##########################################

def delete(file_path):
    try:
        print("delete file")
        os.remove(file_path)
        print(f"File {file_path} deleted successfully.")
    except OSError as e:
        print(f"Error deleting file: {e}")

###############################################

@app.route('/upload', methods=['POST','GET'])
def upload_file():
    if 'file' not in request.files:
        return {'error': 'No file in the request'}, 400

    file = request.files['file']
    if file.filename == '':
        return {'error': 'No file selected'}, 400

    # uploaded file done
    print("upload done")
    print(file.filename)

    # save_path = os.path.join('path', file.filename)
    save_path ='path/' + file.filename
    file.save(save_path)
    print("save done")
    print(save_path)

    # Preprocess the saved audio file
    audio = preprocess_wav_file(save_path)

    # Make a prediction
    prediction = predictmodel(audio)
    print(prediction)
    print("prediction done")
    delete(save_path)


    # Return the prediction as JSON response
    return jsonify({'prediction': prediction})

##################################################

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'file' not in request.files:
        return {'error': 'No file in the request'}, 400

    file = request.files['file']
    if file.filename == '':
        return {'error': 'No file selected'}, 400

    # Save the uploaded file to a desired location
    print("upload ok")
    print(file.filename)

    # save_path = os.path.join('path', file.filename)
    save_path='path/' + file.filename
    file.save(save_path)
    print("save ok")
    print(save_path)

    # Preprocess the saved audio file
    audio = preprocess_wav_file(save_path)

    # Make a prediction
    prediction = predictmodel(audio)
    print(prediction)
    print("prediction done")
    delete(save_path)

    if 'application/json' in request.headers.get('Accept', ''):
        return jsonify({'prediction': prediction})
    else:
        return prediction


    return  render_template('index.html',prediction=prediction,appName="HD prediction")





if __name__ == '__main__':
    app.run(debug=True)