
import sys
# This is a hack to make the AWS server recognize installed packages
sys.path.insert(0, '/opt/python/current/app/conda_env/lib/python3.6/site-packages')

from flask import Flask, request,render_template
from src.absa import ABSA
import json

application = Flask(__name__)

if __name__ != "__main__":
    environment = "production"
else:
    environment = "local"

@application.route('/')
def home():
    return render_template('analyze_reviews.html', environment=environment)


@application.route('/getReviewSentiments', methods=['POST'])
def hello():
    """
    Extract review and desired topic count from the post data and package it
    for a response
    """
    request_data = request.get_json(force=True)
    review = request_data['review']
    topic_count = request_data['topic_count']

    absa = ABSA(review, topic_count)

    topic_results = []
    for i in range(topic_count):
        topic_results.append({
            "topic_name": absa.topic_names[i][1],
            "topic_sentiment": absa.topic_sentiments[i][1]
        })

    return json.dumps(topic_results)


@application.route('/test', methods=['POST'])
def test():
    """
    Test endpoint that returns data in the same format as /getReviewSentiments
    """
    print(request.get_json(force=True))
    return json.dumps([{"topic_name": "Appliance Amazing 715", "topic_sentiment": 5.0}, {"topic_name": "Want Upper Tine", "topic_sentiment": 4.888888888888889}, {"topic_name": "List Handled Bowl", "topic_sentiment": 5.0}, {"topic_name": "Control Run Rinse", "topic_sentiment": 5.0}, {"topic_name": "Left Seldom Flawlessnot", "topic_sentiment": 5.0}, {"topic_name": "Interior Make Plain", "topic_sentiment": 4.8}, {"topic_name": "Score Percent Plate", "topic_sentiment": 5.0}, {"topic_name": "Threw Tackled Everything", "topic_sentiment": 4.666666666666667}, {"topic_name": "Despite Remaining Rigor", "topic_sentiment": 5.0}, {"topic_name": "Oddly Flexibility Shaped", "topic_sentiment": 5.0}])

if __name__ == "__main__":
    application.run()
