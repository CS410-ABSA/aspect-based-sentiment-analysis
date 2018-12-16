# Aspect-Based Sentiment Analysis

The purpose of this repository is to create an aspect-based sentiment analysis (ABSA) model which involves a couple of steps. First, a review or comment is split up into topics using an unsupervised topic clustering algorithm. Then each sentence in the review is assigned to one of the topics and the sentiment for that sentence is evaluated. This allows us to assign an average sentiment for each topic.

## Setting Up The Environment

Please install Anaconda if it isn't already. Next, navigate to the local cloned repository and type the following in the linux shell:

* conda env create -f environment.yml

After the environment has been loaded, activate the environment with the following comand:

* source activate tensorflow

## Training Sentiment Models

To train the sentiment models, simply navigate to the src/training folder and run the following python files:

* setup.py

* train_sent_bow.py

* train_sent_cnn.py

The models can then be tested by running the "test_sentiment.py" file
