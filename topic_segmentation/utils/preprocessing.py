
from nltk import tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from math import log
from numpy import zeros

englishStopwords = stopwords.words('english')
lemma = WordNetLemmatizer()

def preprocessParagraph(paragraph):
    sentences = tokenize.sent_tokenize(paragraph.lower())
    return [ preprocessSentence(sentence) for sentence in sentences ]

def preprocessSentence(sentence):
    tokenizedSentence = tokenize.word_tokenize(sentence)
    lemmatized = [lemma.lemmatize(token) for token in tokenizedSentence]

    noStopwords = [lemma for lemma in lemmatized
                    if lemma not in englishStopwords
                    and len(lemma) > 2
                    and lemma.count("'") != 1]
    noOddChars = [re.sub('[^\w\s]','',word) for word in noStopwords]
    return noOddChars



def docTermWeight(docTermFreq, termDocFreq, docLength, avgDocLength, numDocs, k=10, b=0.5):
    idf = log((numDocs + 1)/termDocFreq)
    normDocLength = 1 - b + b*(docLength/avgDocLength)
    
    return (k+1)*(docTermFreq)*idf/(docTermFreq + k*normDocLength)
    

"""

Apply a TF/IDF/Doc Length Normalization weighting scheme based on that of BM25

Parameters
----------
sentences : list of strings
    A list of sentences
    
Returns
-------
list of lists
    A list of sentences in BOW format with weighting applied
"""
def sentencesToBow(documents):
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
            termWeight = docTermWeight(docTermFreq, termDocFreq, docLength, avgDocLength, numDocs, k=5, b=0.6)
            newBowTerm = (bowTerm[0], termWeight)
            bowSentence.append(newBowTerm)
            
        bowSentences.append(bowSentence)
    
    return bowSentences, dictionary

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
    

        
    