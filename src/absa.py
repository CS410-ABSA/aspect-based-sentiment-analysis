
from .topic_extraction.topic_extractor import TopicExtractor
from .sentiment_prediction.sentiment_prediction import predict_sentiments
from numpy import array, mean
from nltk import tokenize

class ABSA:
    def __init__(self, review, numTopics=5):
        sentences = tokenize.sent_tokenize(review)
        self.topicExtractor = TopicExtractor(sentences, numTopics)
        doc_topics = self.topicExtractor.get_doc_topics()
        doc_senetiments = array(predict_sentiments(sentences))

        topic_sentiments = []
        for i in range(numTopics):
            topic_sentiments.append((i, mean(doc_senetiments[doc_topics == i])))

        print(topic_sentiments)
