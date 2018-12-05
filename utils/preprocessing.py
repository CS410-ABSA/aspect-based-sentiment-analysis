
from nltk import tokenize
from nltk.stem.wordnet import WordNetLemmatizer

def preprocess(paragraph):
    sentences = tokenize.sent_tokenize(paragraph.lower())
    lemma = WordNetLemmatizer()
    clean_sentences = []
    for sentence in sentences:
        sentence = re.sub('[^\w\s]','',sentence)
        lemma_sent = []
        for word in sentence.split():
            word = lemma.lemmatize(word)
            lemma_sent.append(word)
        clean_sentences.append(' '.join(lemma_sent))
    return clean_sentences
