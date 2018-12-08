
from utils.preprocessing import preprocessParagraph
import sys
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath
from nltk import tokenize

review = open('review.txt', 'r').read()

sentences = preprocessParagraph(review)

dictionary = Dictionary(sentences)

# scikit learn: tdf vectorizor
# also try to use idf
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]

lda = LdaModel(corpus, num_topics=10, chunksize=20, passes=50, id2word=dictionary)

topics = lda.print_topics(num_words=4)

doc_scores = [lda.get_document_topics(document) for document in corpus]

def max_by_index(array, index):
    return max(array, key=lambda item: item[index])


doc_topics = [max_by_index(doc_score,1) for doc_score in doc_scores]

raw_sentences = tokenize.sent_tokenize(review)

first_topic_sentences = [raw_sentences[i] for i,tuple_ in enumerate(doc_topics)
                            if tuple_[0] == 1]