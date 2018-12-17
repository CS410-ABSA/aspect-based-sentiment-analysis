
from nltk import tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from math import log
from numpy import zeros

englishStopwords = stopwords.words('english')
lemma = WordNetLemmatizer()


def preprocessSentences(sentences):
    return [ preprocessSentence(sentence) for sentence in sentences ]


def preprocessSentence(sentence):
    """
    For each sentence
        - lemmatize
        - remove stopwords
        - remove short words
        - remove single quotes
        - remove non standard characters
    """
    tokenizedSentence = tokenize.word_tokenize(sentence.lower())
    lemmatized = [lemma.lemmatize(token) for token in tokenizedSentence]

    noStopwords = [lemma for lemma in lemmatized
                    if lemma not in englishStopwords
                    and len(lemma) > 2
                    and lemma.count("'") != 1]
    noOddChars = [re.sub('[^\w\s]','',word) for word in noStopwords]
    return noOddChars


def docTermWeight(docTermFreq, termDocFreq, docLength, avgDocLength, numDocs, k=10, b=0.5):
    """Calculate term weighting

    Calculate term weighting using TF-IDF term weighting as well as document length
    normalization
    """
    idf = log((numDocs + 1)/termDocFreq)
    normDocLength = 1 - b + b*(docLength/avgDocLength)

    return (k+1)*(docTermFreq)*idf/(docTermFreq + k*normDocLength)


def sentencesToBow(documents):
    """Convert given tokenized sentences into a weighted bag-of-words format

    Create a dictionary from the given sentences and find global statistics.
    Then for each document calculate document statistics and find the BOW
    representation for the document. For each BOW term calculate term statistics
    and find the term weigth using all the gathered statistics.

    Parameters
    ----------
    documents : list-of-lists
        An array of sentences that have been tokenized

    Returns
    -------
    bowSentences: list-of-lists
        An array of sentences in weighted BOW format: (term_id, term_weight)
    """
    bowSentences = []
    dictionary = Dictionary(documents)

    numDocs = len(documents);
    avgDocLength = sum(map(len, documents))/numDocs
    for document in documents:
        bowSentence = []
        docLength = len(document)
        bowDoc = dictionary.doc2bow(document)
        for bowTerm in bowDoc:

            docTermFreq = bowTerm[1]
            termDocFreq = dictionary.dfs[bowTerm[0]]

            termWeight = docTermWeight(docTermFreq, termDocFreq, docLength, avgDocLength, numDocs, k=5, b=0.1)
            newBowTerm = (bowTerm[0], termWeight)
            bowSentence.append(newBowTerm)

        bowSentences.append(bowSentence)

    return bowSentences, dictionary


"""
Experimental function for creating a TDF matrix from a list of documents.
"""
def createTdf(bowDocs, numTerms):
    tdf = []

    for bowDoc in bowDocs:
        indecies = []
        weights = []
        for termTuple in bowDoc:
            indecies.append(termTuple[0])
            weights.append(termTuple[1])
        zeroArray = zeros(numTerms)
        zeroArray[indecies] = weights
        tdf.append(zeroArray)

    return tdf;
