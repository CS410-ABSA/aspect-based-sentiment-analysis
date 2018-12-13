
from src.absa import ABSA

def test_absa():
    review = open('test/resources/data/review2.txt', 'r').read()
    absa = ABSA(review)
    assert(absa != None)
