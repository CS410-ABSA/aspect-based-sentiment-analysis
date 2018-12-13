
from src.topic_extraction.topic_extractor import TopicExtractor
from nltk import tokenize

def test_extract_topics():
    review = open('test/resources/data/review2.txt', 'r').read()
    rawSentences = tokenize.sent_tokenize(review)
    topicExtractor = TopicExtractor(rawSentences, num_topics=2)

    print(topicExtractor.get_sentences_by_topic_id(0, processed=True))

    assert(topicExtractor != None)
