# Sentimental-analysis-using-CNN-and-HNN-
Sentiment Analysis using CNN and HAN Models
This project implements sentiment analysis using two different deep learning models:

Convolutional Neural Network (CNN): A simple CNN model to classify sentiment based on text input.
Hierarchical Attention Network (HAN): A more advanced model that uses attention mechanisms to focus on important words for better sentiment prediction.
Both models are trained on the NLTK Movie Reviews dataset, which contains positive and negative movie reviews. The models are trained to predict the sentiment of a given movie review as either Positive or Negative.

Project Structure
cnn_sentiment_model.h5: Trained CNN model for sentiment analysis.
han_sentiment_model.h5: Trained HAN model for sentiment analysis.
tokenizer.pkl: Tokenizer used for processing text data.
sentiment_analysis.py: Python script that loads the models, tokenizes input text, and performs sentiment prediction.
README.md: This file.
Requirements
To run this project, you'll need the following dependencies:

TensorFlow
Keras
NLTK
Gradio
scikit-learn
numpy
pandas
You can install these dependencies by running the following:

bash
Copy
Edit
pip install tensorflow nltk gradio scikit-learn numpy pandas
Model Training
To train the models:

Data Preparation: The NLTK Movie Reviews dataset is used. It's loaded and preprocessed into tokenized sequences.
Model Building: Two models are built using Keras:
CNN Model: A simple Convolutional Neural Network with an embedding layer, convolution layer, and a dense output layer.
HAN Model: A Hierarchical Attention Network that uses LSTM and attention mechanisms.
Training: Both models are trained using the binary cross-entropy loss function and the Adam optimizer.
After training, the models are saved to disk for future inference.

Sentiment Prediction
To use the models for prediction:

Load the models and the tokenizer.
Preprocess the input text to convert it into a format suitable for the models.
Use the models to predict the sentiment of the text.
Display the results: Each model provides a probability of the sentiment being Positive or Negative.
Gradio Interface
A Gradio interface is provided to interact with the models through a simple web interface. You can input any text and get the sentiment predictions from both models.

To launch the Gradio interface:

python
interface = gr.Interface(fn=predict_sentiment, inputs="text", outputs="text")
interface.launch(share=True)
This will open a local or publicly accessible link where you can input text and receive predictions from both the CNN and HAN models.

python
test_text = "The movie was amazing!"
predict_sentiment(test_text)
Expected Output:
plaintext
CNN Model Prediction: Positive (0.9062)
HAN Model Prediction: Positive (0.9305)
