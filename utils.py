import pandas as pd
import numpy as np
import ibm_boto3
from ibm_botocore.client import Config
from tempfile import NamedTemporaryFile

# initialize a cloud object store
def get_cloud_object_store(cos_arguments):
    # Open connection to cloud object storage
    cos = ibm_boto3.client("s3",
                           ibm_api_key_id=cos_arguments["apikey"],
                           ibm_service_instance_id=cos_arguments["resource_instance_id"],
                           ibm_auth_endpoint="https://iam.ng.bluemix.net/oidc/token",
                           config=Config(signature_version="oauth"),
                           endpoint_url="https://s3.us-south.objectstorage.softlayer.net")

    return cos

def dwnld_csvdf_cos(cos, bucket, key):
    # the temporary in-memory file for holding the model file
    tmpfile = NamedTemporaryFile()

    # read the model into this file
    cos.download_file(bucket, key, tmpfile.name)

    # convert to df and return
    return pd.read_csv(tmpfile.name)

def get_topic_stats(x):
    count = x.shape[0]
    mean_sentiment = np.mean(x['watson_sentiment'])
    mean_emotion_anger = np.mean(x['watson_emotion_anger'])
    mean_emotion_disgust = np.mean(x['watson_emotion_disgust'])
    mean_emotion_fear = np.mean(x['watson_emotion_fear'])
    mean_emotion_joy = np.mean(x['watson_emotion_joy'])
    mean_emotion_sadness = np.mean(x['watson_emotion_sadness'])

    columns = ['count',
               'mean_sentiment',
               'mean_emotion_anger',
               'mean_emotion_disgust',
               'mean_emotion_fear',
               'mean_emotion_joy',
               'mean_emotion_sadness']

    stats = pd.DataFrame(columns=columns)
    stats.loc[0] = [count,
                    mean_sentiment,
                    mean_emotion_anger,
                    mean_emotion_disgust,
                    mean_emotion_fear,
                    mean_emotion_joy,
                    mean_emotion_sadness]

    return stats

def get_sentiment_string(row):
    sentiment_score = row.watson_sentiment
    sentiment = "Neutral"
    if sentiment_score >= 0.3:
        sentiment = "Positive"
    elif sentiment_score <= -0.3:
        sentiment = "Negative"

    sentiment_str = "<i>%s</i> (%.2f)" % (sentiment, sentiment_score)
    return (sentiment_str)


def get_top_emotion_string(row):
    emotions = row[['watson_emotion_anger', 'watson_emotion_disgust', 'watson_emotion_fear', 'watson_emotion_joy',
                    'watson_emotion_sadness']]
    max_emotion = max(emotions)
    emotion_str = 'None'
    if row.watson_emotion_anger == max_emotion:
        emotion_str = '<i>anger</i> (%.2f)' % max_emotion
    elif row.watson_emotion_disgust == max_emotion:
        emotion_str = '<i>disgust</i> (%.2f)' % max_emotion
    elif row.watson_emotion_fear == max_emotion:
        emotion_str = '<i>fear</i> (%.2f)' % max_emotion
    elif row.watson_emotion_joy == max_emotion:
        emotion_str = '<i>joy</i> (%.2f)' % max_emotion
    elif row.watson_emotion_sadness == max_emotion:
        emotion_str = '<i>sadness</i> (%.2f)' % max_emotion

    return (emotion_str)
