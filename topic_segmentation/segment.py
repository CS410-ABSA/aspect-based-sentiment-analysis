
from .utils import preprocessSentences, sentencesToBow
from gensim.models.ldamodel import LdaModel

def max_by_index(array, index):
    return max(array, key=lambda item: item[index])

def extract_topics(rawSentences, num_topics=5):
    sentences = preprocessSentences(rawSentences)

    sentBows, dictionary = sentencesToBow(sentences)

    for sentBow in sentBows:
        print("-------------------------------")
        for bowToken in sentBow:
            print(dictionary[bowToken[0]] + " => " + str(bowToken[1]))

    topic_model = LdaModel(sentBows, num_topics=num_topics, chunksize=len(sentBows), passes=30, id2word=dictionary, update_every=0, alpha='auto')

    doc_scores = [topic_model.get_document_topics(document) for document in sentBows]

    doc_topics = [max_by_index(doc_score,1) for doc_score in doc_scores]



#    raw_sentences = tokenize.sent_tokenize(review)
#
#    first_topic_sentences = [raw_sentences[i] for i,tuple_ in enumerate(doc_topics)
#                                if tuple_[0] == 1]

    return doc_topics, topic_model
