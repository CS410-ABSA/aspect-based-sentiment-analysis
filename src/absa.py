
from .topic_extraction.topic_extractor import TopicExtractor
from .sentiment_prediction.sentiment_prediction import predict_sentiments
from numpy import array, mean
from nltk import tokenize

class ABSA:
    """
    Class merging usage of sentiment extraction and topic segmentation

    Attributes
    ----------
    topic_extractor: TopicExtractor
        Topic extractor created from the given review
    topic_sentiments: list of tuples
        List of tuples (topic_id, topic_sentiment)
    topic_names: list of tuples
        List of tuples (topic_id, topic_name)
    """

    def __init__(self, review, num_topics=5):
        """
        Tokenize the review into sentences, extract the sentences for each topic,
        find sentiments for each sentence, and find the sentiment mean across
        all sentences in each topic. Also extract the topic names.

        Parameters
        ---------
        review: string
            raw review
        num_topics: integer, optional
            desired number of topics to extract
        """
        sentences = tokenize.sent_tokenize(review)
        self.topic_extractor = TopicExtractor(sentences, num_topics)
        doc_topics = self.topic_extractor.get_doc_topics()
        doc_senetiments = array(predict_sentiments(sentences))

        self.topic_sentiments = [(i, mean(doc_senetiments[doc_topics == i])) for i in range(numTopics)]

        self.topic_names = self.topic_extractor.get_topic_names()
