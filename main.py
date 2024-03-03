from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import cv2
import os
import pickle
import pyttsx3

# Load the trained model and tokenizer
model_path = 'new_final_model.h5'  # Replace with the actual path
tokenizer_path = 'tokenizer.pkl'  # Replace with the actual path

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model(model_path)

# Define max length (you may need to set this based on your training)
max_length = 35

# Function to extract features using VGG16
def extract_features(filename):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image = cv2.imread(filename)
    image = cv2.resize(image, (224, 224))
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

# Function to generate caption
def generate_caption(image_filename):
    # load the image
    image = Image.open(image_filename)

    # Extract features
    photo = extract_features(image_filename)

    # Initialize in_text with the start sequence
    in_text = 'startseq'

    # Generate caption word by word
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        if word == 'endseq':
            break
        in_text += ' ' + word

    # Display the generated caption (excluding startseq and endseq)
    generated_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    print('Generated Caption:', generated_caption)

    # Speak out the caption
    speak_caption(generated_caption)

    # Display the image
    plt.imshow(image)
    plt.show()

# Add the idx_to_word function here
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to use TTS to speak out the given text
def speak_caption(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Replace with the actual path to the image
image_filename = 'free-images.jpg'
generate_caption(image_filename)
