
from src.topic_extraction.topic_extractor import TopicExtractor
from src.topic_extraction.keyword_extractor import KeywordExtractor
from nltk import tokenize

def test_extract_topics():
    NUM_TOPICS = 10

    review = open('test/resources/data/review2.txt', 'r').read()
    rawSentences = tokenize.sent_tokenize(review)
    topicExtractor = TopicExtractor(rawSentences, num_topics=NUM_TOPICS)

    assert(topicExtractor != None)
    assert(len(topicExtractor.get_topic_names()) == NUM_TOPICS)
