
from utils.preprocessing import preprocessParagraph, sentencesToBow, createTdf
from gensim.models.ldamodel import LdaModel
from nltk import tokenize
from numpy.linalg import svd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


review = open('review2.txt', 'r').read()

sentences = preprocessParagraph(review)

sentBow, dictionary = sentencesToBow(sentences)

tdf= createTdf(sentBow, len(dictionary.token2id))

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(tdf)

U = svd(tdf)[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(U[:,0], U[:,1],c='r', marker='o')

plt.show()

lda = LdaModel(sentBow, num_topics=6, chunksize=len(sentBow), passes=30, id2word=dictionary, update_every=0, alpha='auto')

topics = lda.print_topics(num_words=4)

doc_scores = [lda.get_document_topics(document) for document in sentBow]

def max_by_index(array, index):
    return max(array, key=lambda item: item[index])


doc_topics = [max_by_index(doc_score,1) for doc_score in doc_scores]

raw_sentences = tokenize.sent_tokenize(review)

first_topic_sentences = [raw_sentences[i] for i,tuple_ in enumerate(doc_topics)
                            if tuple_[0] == 1]


