import urllib.request
import gzip
import shutil
import os

print("Retrieving 1")
urllib.request.urlretrieve("https://s3.amazonaws.com/absa-models/bin/bow_sa_pipeline.joblib", "../../models/bow_sa_pipeline.joblib")
print("Retrieving 2")
urllib.request.urlretrieve("https://s3.amazonaws.com/absa-models/bin/cnn_absa_model.h5", "../../models/cnn_absa_model.h5")
print("Retrieving 3")
urllib.request.urlretrieve("https://s3.amazonaws.com/absa-models/bin/cnn_label_binarizer.joblib", "../../models/cnn_label_binarizer.joblib")
print("Retrieving 4")
urllib.request.urlretrieve("https://s3.amazonaws.com/absa-models/bin/word2vec.model", "../../models/word2vec.model")
print("Retrieving 5")
urllib.request.urlretrieve("https://s3.amazonaws.com/absa-models/bin/word2vec.model.trainables.syn1neg.npy", "../../models/word2vec.model.trainables.syn1neg.npy")
print("Retrieving 6")
urllib.request.urlretrieve("https://s3.amazonaws.com/absa-models/bin/word2vec.model.wv.vectors.npy", "../../models/word2vec.model.wv.vectors.npy")

# with gzip.open('amazon_reviews_us_Electronics_v1_00.tsv.gz', 'rb') as f_in:
#     with open('amazon_reviews_us_Electronics_v1_00.tsv', 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out)
#
#
# if os.path.exists("amazon_reviews_us_Electronics_v1_00.tsv.gz"):
#     os.remove("amazon_reviews_us_Electronics_v1_00.tsv.gz")
# else:
#     print("The file does not exist")
    