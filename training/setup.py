import urllib.request
import gzip
import shutil
import os

urllib.request.urlretrieve("https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Electronics_v1_00.tsv.gz", "amazon_reviews_us_Electronics_v1_00.tsv.gz")

with gzip.open('amazon_reviews_us_Electronics_v1_00.tsv.gz', 'rb') as f_in:
    with open('amazon_reviews_us_Electronics_v1_00.tsv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


if os.path.exists("amazon_reviews_us_Electronics_v1_00.tsv.gz"):
    os.remove("amazon_reviews_us_Electronics_v1_00.tsv.gz")
else:
    print("The file does not exist")
