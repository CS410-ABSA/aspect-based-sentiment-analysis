
from src.absa import ABSA

def test_absa():
    review = open('test/resources/data/review2.txt', 'r').read()
    absa = ABSA(review, 5)

    assert(len(absa.topic_sentiments) == 5)
    assert(len(absa.topic_names) == 5)
    assert(absa != None)
