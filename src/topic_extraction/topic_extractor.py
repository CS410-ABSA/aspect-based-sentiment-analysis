
from .utils import preprocessSentences, sentencesToBow
from gensim.models.ldamodel import LdaModel
from numpy import array

def max_by_index(array, index):
    """Helper function for finding the list in a list of lists that achieves
       the highest value of all lists at the given index

    Parameters
    ---------
    array: List of lists
        list to be processed
    index: integer
        Index of values in each element that should be compared

    Returns
    -------
    list
        list that has the highest value of all lists at the given index
    """
    return max(array, key=lambda item: item[index])

class TopicExtractor:
    """
    Class used to extract topics from a list of sentences

    Attributes
    ----------
    num_topics: integer
        Desired numer of topics to extract
    raw_sentences: list of strings
        List of sentences originally passed to the class
    processed_sentences: list of lists of strings
        List of processed and tokenized sentences
    sent_bows: List of list of tuples
        List of sentences in BOW format where each tuple is (term_id, term_weight)
    dictionary: gensim.corpora.Dictionary
        Gensim dictionary produced by .utils.sentencesToBow
    topic_model: gensim.models.ldamodel.LdaModel
        Gensim LDA model produced from the given sentences
    doc_topics: List of integers
        List where the ith element is the topic of the ith sentence passed to the constructor

    Methods
    -------
    get_doc_topics()
        Returns self.doc_topics
    get_topic_names()
        Returns topic names for each sentence
    print_token_weights()
        Helper function for printing weights of each term in each sentence
    get_sentences_by_topic_id(topic_id, processed=False)
        Helper function for retrieving sentences for the given topic (processed or not processed)
    """

    def __init__(self, raw_sentences, num_topics):
        """
        Preprocess sentences, convert them to BOW format, and then pass
        the BOW sentences to LdaModel. Get the topic distribution for each
        sentence and for each sentence find the topic with the highest weight.
        Finally, extract the topic index into self.doc_topics.

        Parameters
        ----------
        raw_sentences: list of strings
            A list of unprocessed sentences
        num_topics: integer
            Desired numer of topics to extract
        """
        self.num_topics = num_topics
        self.raw_sentences = array(raw_sentences)
        self.processed_sentences = preprocessSentences(raw_sentences)
        self.sent_bows, self.dictionary = sentencesToBow(self.processed_sentences)

        self.topic_model = LdaModel(self.sent_bows,
                                    num_topics=num_topics,
                                    chunksize=len(self.sent_bows),
                                    passes=30,
                                    id2word=self.dictionary,
                                    update_every=0,
                                    alpha='auto')

        doc_scores = [self.topic_model.get_document_topics(document) for document in self.sent_bows]
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

    """Determine human-readable labels for each topic

    For each topic, pull out the top three words that contributed most
    to the topic. Join them with a space and send them to title case.

    Returns
    -------
    list of tuples
        A list of tuples where each tuple is (topic_id, topic_name)
    """
    def get_topic_names(self):
        topic_representations = self.topic_model.show_topics(num_topics=self.num_topics, num_words=3, formatted=False)

        topic_names = []
        for topic_words_tuple in topic_representations:
            topic_name = " ".join(map(lambda x: x[0], topic_words_tuple[1])).title()
            topic_names.append((topic_words_tuple[0], topic_name))

        return topic_names


    def print_token_weights(self):
        for sentBow in self.sent_bows:
            print("-------------------------------")
            for bowToken in sentBow:
                print(self.dictionary[bowToken[0]] + " => " + str(bowToken[1]))


    def get_sentences_by_topic_id(self, topic_id, processed=False):
        if topic_id + 1 > self.num_topics:
            raise Exception("Invalid topic id: " + str(topic_id))

        if processed:
            return array(self.processed_sentences)[self.doc_topics == topic_id]

        return self.raw_sentences[self.doc_topics == topic_id]
