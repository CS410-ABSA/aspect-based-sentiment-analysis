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

## Running Web Interface Locally
To run the server locally and test out the algorithm, first run `python setup.py` (this will probably take a while as it has to download several model files to the `./models` directory).

!!! NOTE: Part of what setup.py does is download some nltk data to `/usr/local/lib/nltk_data`. Make sure to delete this directory if you don't want it.!!!

Finally run `python api.py` and go to http://localhost:5000.

# Methodology

This project comprises of four main modules: A topic extraction module, a sentiment prediction module, an ABSA module to tie the sentiment and topic modules together, and a web-interface/REST API module to expose the algorithm publicly

## Sentiment Prediction Module
### Model Training
(Code is in `src/training`)

To evaluate the sentiment of each sentence in the review we trained two separate models and averaged their scores. Both models were trained on Amazon reviews with the target variable being the rating (1-5) that the customer gave for the product. The first model is a bag-of-words model and the code is in: `train_sent_bow.py`. This model tokenizes each sentence into a TF-IDF weighed bag of words and uses logistic regression to model word relationships to output scores. The second model uses a deep-learning convolution neural network (CNN). Each word is vectorized in 150-dimensional space using the gensim Word2Vec library. Each associated word vector is stacked on top of each other in the order that word appears in the sentence. The maximum number of words we allowed for each sentence was 50 resulting in each sentence being converted into a 150 by 50 matrix. So if a sentence had fewer than 50 words, the empty space would be converted to zero-vectors. If the sentence had more than 50 words, it was simply truncated. These input sentence matrices were then fed through a convolutional neural network and trained against the target variables to learn what word patterns resulted in positive/negative scores. After the model is trained, it is saved to the director so that it can be used for sentiment evaluation on new reviews. 

### Sentiment Inference
(Code is in `src/sentiment_prediction`)

This code simply imports the pre-trained models, preprocesses the incoming text, and runs sentiment prediction.

## Topic Extraction Module

### Topic Segmentation
(Code is in `src/topic_segmentation`)

In order to segment a review into topics, the constructor of the class `TopicExtractor` in `src/topic_extraction/topic_extraction.py` takes as input an array of unprocessed sentences and the desired number of topics to extract. The TopicExtractor first preprocesses the sentences and then converts those sentences into a Bag Of Words format with TF-IDF and document length normalization being applied to each term. This bag of words is then passed to the class `gensim.models.ldamodel.LdaModel`. Then the topic weighting for each sentence is found by passing the BOW format to the method `get_document_topics` of the gensim topic model. Finally, the topic of each sentence is inferred by choosing the topic with the highest weight in the sentence. The method `TopicExtractor::get_doc_topics` can then be used to retrieve the topic of each sentence.

More implementation details can be found in the source code.

### Inferring a Topic Name
(Code is in `src/topic_segmentation`)

In order to infer a human-readable label of each topic, we used the top three words that contributed most to each topic. Once a `TopicExtractor` object has been created, the `TopicExtractor::get_topic_names` method can be used to retrieve topic labels for each topic.

## ABSA Module
(Code is in `src/absa.py`)

In order to tie together the two algorithms, the class `ABSA` in `src/absa.py` finds the sentences for each topic and the sentiments for each sentence. Then sentiments are averaged over all sentences in a topic to find an average sentiment for that topic. The topic names are also retrieved using `TopicExtractor::get_topic_names`.

## Web Interface/REST API Module
(Code is in `api.py` and `templates`)

The root route for the api returns an html page defined in `templates/analyze_reviews.html`. This web-page calls the `/getReviewSentiments` endpoint, passing the user-provided review and topic count. Then `/getReviewSentiments` passes the request and topic count to the `ABSA` class to collect and return topic sentiments and names.
