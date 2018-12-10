
from .utils import preprocessSentences, sentencesToBow
from gensim.models.ldamodel import LdaModel

def max_by_index(array, index):
    return max(array, key=lambda item: item[index])

class TopicExtractor:
    def __init__(self, rawSentences, num_topics):     
        sentences = preprocessSentences(rawSentences)
        self.sentBows, self.dictionary = sentencesToBow(sentences)

        self.topic_model = LdaModel(self.sentBows,
                                    num_topics=num_topics,
                                    chunksize=len(self.sentBows),
                                    passes=30,
                                    id2word=self.dictionary,
                                    update_every=0,
                                    alpha='auto')

        doc_scores = [self.topic_model.get_document_topics(document) for document in self.sentBows]
        self.doc_topics = [max_by_index(doc_score,1) for doc_score in doc_scores]

    def print_token_weights(self):
        for sentBow in self.sentBows:
            print("-------------------------------")
            for bowToken in sentBow:
                print(self.dictionary[bowToken[0]] + " => " + str(bowToken[1]))
