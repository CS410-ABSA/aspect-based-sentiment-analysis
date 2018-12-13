
from src.sentiment_prediction.sentiment_prediction import predict_sentiments

def test_predict_sentiments():
    sentences = ['this does not work', 'this is an ok product']
    predictions = predict_sentiments(sentences)

    print(predictions)
