Here is a `README.md` file for your project:

```markdown
# Sentiment Analysis using CNN and HAN Models

This project demonstrates sentiment analysis on movie reviews using two different deep learning models: Convolutional Neural Network (CNN) and Hierarchical Attention Network (HAN). The models are trained on the NLTK movie reviews dataset and can predict whether a given review is positive or negative.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
  - [CNN Model](#cnn-model)
  - [HAN Model](#han-model)
- [Gradio Interface](#gradio-interface)
- [License](#license)

## Installation

To run this project, you need to install the required dependencies. You can do this by running the following command:

```bash
pip install tensorflow gradio nltk scikit-learn pandas h5py
```

Additionally, you need to download the NLTK movie reviews dataset:

```python
import nltk
nltk.download('movie_reviews')
```

## Usage

1. **Data Preparation**: The dataset is loaded and preprocessed using tokenization and padding.
2. **Model Training**: The CNN and HAN models are trained on the preprocessed data.
3. **Prediction**: The trained models can be used to predict the sentiment of new reviews.
4. **Gradio Interface**: A web-based interface is provided to interact with the models and get predictions.

### Running the Notebook

1. Open the Jupyter notebook `sentimental_analysis_of_cnn_and_hnn_.ipynb`.
2. Run each cell sequentially to load the data, train the models, and make predictions.
3. Use the Gradio interface to input text and get sentiment predictions.

## Models

### CNN Model

The CNN model consists of the following layers:
- **Embedding Layer**: Converts words into dense vectors.
- **Conv1D Layer**: Applies convolutional filters to extract features.
- **GlobalMaxPooling1D Layer**: Reduces the dimensionality of the output.
- **Dense Layers**: Fully connected layers for classification.

### HAN Model

The HAN model uses a hierarchical attention mechanism to focus on important parts of the text. It consists of:
- **Embedding Layer**: Converts words into dense vectors.
- **Bidirectional LSTM**: Captures contextual information from the text.
- **Attention Layer**: Weights the importance of different parts of the text.
- **Dense Layer**: Output layer for classification.

## Gradio Interface

The project includes a Gradio interface that allows users to input text and get sentiment predictions from both the CNN and HAN models. The interface can be launched using the following code:

```python
interface = gr.Interface(fn=predict_sentiment, inputs="text", outputs="text")
interface.launch(share=True)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
```

### Explanation:
- **Installation**: Lists the required dependencies and how to install them.
- **Usage**: Provides a step-by-step guide on how to run the notebook and use the models.
- **Models**: Describes the architecture of the CNN and HAN models.
- **Gradio Interface**: Explains how to use the Gradio interface for interactive predictions.
- **License**: Specifies the license under which the project is distributed.

You can save this content in a file named `README.md` in your project directory.
