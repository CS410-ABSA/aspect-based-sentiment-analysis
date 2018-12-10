

from topic_segmentation.segment import TopicExtractor
from nltk import tokenize

def test_extract_topics():
    review = open('../review2.txt', 'r').read()
    rawSentences = tokenize.sent_tokenize(review)
    topicExtractor = TopicExtractor(rawSentences, num_topics=5)

    topicExtractor.print_token_weights()

if __name__ == "__main__":
    test_extract_topics()
else:
    raise Exception('Not a module!')
