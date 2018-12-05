
from nltk import tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
from nltk.corpus import stopwords

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
