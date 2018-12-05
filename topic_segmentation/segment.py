
from utils.preprocessing import preprocessParagraph
import sys
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath

review = open('topic_segmentation/review.txt', 'r')

sentences = preprocessParagraph(review.read())

dictionary = Dictionary(sentences)
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]

lda = LdaModel(corpus, num_topics=5, passes=15, id2word=dictionary)
# temp_file = datapath("model")
# lda.save(temp_file)


# lda = LdaModel.load(temp_file)
topics = lda.print_topics(num_words=4)
for topic in topics:
    print(topic)
