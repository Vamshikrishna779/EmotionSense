from flask import Flask, render_template, request, jsonify  # Import jsonify
import librosa
import numpy as np
import joblib
import os
# Initialize Flask app
#app = Flask(__name__)

# Define allowed file types
#ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'}

# Load the pre-trained model
model = joblib.load('EmotionEnsembleClassifier.joblib')

# Initialize Flask app


# Emotion labels mapping (adjust this to your actual label names)
emotion_map = {
    'OAF_Fear': 'Fear',
    'OAF_Pleasant_surprise': 'Pleasant Surprise',
    'OAF_Sad': 'Sad',
    'OAF_angry': 'Angry',
    'OAF_disgust': 'Disgust',
    'OAF_happy': 'Happy',
    'OAF_neutral': 'Neutral'
}

# Initialize Flask app
app = Flask(__name__)

# Function to extract MFCC features from an audio file
def extract_mfcc(audio_path, sample_rate=16000, n_mfcc=13):
    try:
        # Load the audio file
        audio_data, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        
        # Return the mean of each MFCC coefficient across time steps
        return np.mean(mfcc, axis=1)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Route for rendering the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for predicting the emotion from an uploaded audio file
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    audio_file = request.files['audio']

    # Save the uploaded file temporarily
    audio_path = 'temp_audio.wav'
    audio_file.save(audio_path)

    # Extract features from the uploaded audio file
    mfcc = extract_mfcc(audio_path)
    
    if mfcc is None:
        return jsonify({'error': 'Failed to process audio'}), 400

    # Make prediction
    prediction = model.predict([mfcc])

    # Get the predicted emotion
    predicted_emotion = prediction[0]
    
    # Map folder name to readable emotion
    predicted_emotion = emotion_map.get(predicted_emotion, 'Unknown')

    # Remove the temporary audio file
    os.remove(audio_path)

    return jsonify({'emotion': predicted_emotion})

if __name__ == '__main__':
    app.run(debug=True)