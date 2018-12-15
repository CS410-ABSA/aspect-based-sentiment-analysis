
from .utils import preprocessSentences, sentencesToBow
from gensim.models.ldamodel import LdaModel
from numpy import array

def max_by_index(array, index):
    return max(array, key=lambda item: item[index])

class TopicExtractor:
    def __init__(self, raw_sentences, num_topics):
        self.num_topics = num_topics
        self.raw_sentences = array(raw_sentences)
        self.processed_sentences = preprocessSentences(raw_sentences)
        self.sentBows, self.dictionary = sentencesToBow(self.processed_sentences)

        self.topic_model = LdaModel(self.sentBows,
                                    num_topics=num_topics,
                                    chunksize=len(self.sentBows),
                                    passes=30,
                                    id2word=self.dictionary,
                                    update_every=0,
                                    alpha='auto')

        doc_scores = [self.topic_model.get_document_topics(document) for document in self.sentBows]
        max_doc_topic_scores = [max_by_index(doc_score,1) for doc_score in doc_scores]
        self.doc_topics = array(list(map(lambda t:t[0], max_doc_topic_scores)))

    """
    Returns
    -------
    doc_topics: array
        array of topic ids where the i-th element is the topic for
        the i-th sentence in self.raw_sentences/self.processed_sentences
    """
    def get_doc_topics(self):
        return self.doc_topics


    def print_token_weights(self):
        for sentBow in self.sentBows:
            print("-------------------------------")
            for bowToken in sentBow:
                print(self.dictionary[bowToken[0]] + " => " + str(bowToken[1]))


    def get_topic_names(self):
        topic_representations = self.topic_model.show_topics(num_topics=self.num_topics, num_words=3, formatted=False)

        topic_names = []
        for topic_words_tuple in topic_representations:
            topic_name = " ".join(map(lambda x: x[0], topic_words_tuple[1])).title()
            topic_names.append((topic_words_tuple[0], topic_name))

        return topic_names


    def get_sentences_by_topic_id(self, topic_id, processed=False):
        if topic_id + 1 > self.num_topics:
            raise Exception("Invalid topic id: " + str(topic_id))

        if processed:
            return array(self.processed_sentences)[self.doc_topics == topic_id]

        return self.raw_sentences[self.doc_topics == topic_id]
