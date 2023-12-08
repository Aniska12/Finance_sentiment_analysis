# Finance_sentiment_analysis

# Project Overview

This project involves the fusion of two different models: Graph Convolutional Network (GCN) and Long Short-Term Memory (LSTM). The goal is to perform attention fusion of these two models for sentiment analysis on financial datasets.

## Model Fusion

The fusion process combines the strengths of GCN, which excels at capturing graph-based dependencies, and LSTM, known for its ability to capture sequential patterns. The attention fusion mechanism enhances the overall sentiment analysis capabilities by leveraging the complementary features of both models.

## Sentiment Analysis on Financial Datasets

The fused model is applied to perform sentiment analysis specifically tailored for financial datasets. Analyzing sentiments in financial data is crucial for understanding market trends, investor sentiment, and making informed decisions in the financial domain.

## Requirement
- requirement.txt has all requirements, run the following code
    ```bash
    pip install -r requirement.txt
    ```
- GloVe pre-trained word vectors:
  - Download pre-trained word vectors [here](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors).
  - Put [glove.840B.300d.txt](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) into the `dataset/glove/` folder.  



# Project Instructions

## Vocabulary Generation

To generate the vocabulary for the given datasets, utilize the `build_vocab.sh` file. Execute the following command in bash:

```bash
sh build_vocab.sh
```

# Model Training and Testing

To train the model, execute the following command in your terminal:
```bash
sh train.sh
```
Feel free to modify the train.sh script to experiment with different models and parameters.

To train the model, execute the following command in your terminal:
```bash
sh test.sh
```
Modify the train.sh script to test the specific model.

**The best model will be saved in the bestmodel directory.**


## **Important Note** ##

    - This application is built for educational purposes. 
    - Feel free to explore and modify the code to suit your needs. 
    - If you encounter any issues or have suggestions for improvements, please open an issue on this repository.

