
from topic_segmentation.segment import TopicExtractor
from nltk import tokenize

class ABSA:
    def __init__(self, review, numTopics=5):
        sentences = tokenize.sent_tokenize(review)
        self.topicExtractor = TopicExtractor(sentences, numTopics)
        # self.
