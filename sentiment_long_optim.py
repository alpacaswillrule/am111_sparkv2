'''
Optimized version of sentiment_run. Should work better on larger datsets. However, I would not recommend choosing more
than 300 wacrcs to analyze, as it will take a long time to run. 

For this script to work, a model needs to be in the models folder and it needs to be named model_dl. Training a model is advanced. 
If you want to train a model, please use the training_fin_classifier.py script, but you can also ask me and I will share the model
on google drive. (too large to upload to github)

Usuage:
0. Start a cluster with spark-dev cells
1. run sentiment_long_boot with ip as parameter and pem location and seconr parameter
example: sh sentiment_long_boot.sh ec2-54-160-226-58.compute-1.amazonaws.com "/home/johan/Johan_key.pem"
2.ssh onto master node and mkdir articlespar.parquet, then sudo docker build -t sentiment_long_boot . and run the docker build 
with sudo docker run -e NUMWARCS=120 -e NUMRECORDS=200 -e PYTHONFILETORUN=./sentiment_long_optim.py sentiment_long_boot 
if running locally just build and run dockerfile but with additional arguments of (-e AWS_ACCESS_KEY_ID= -e AWS_SECRET_ACCESS_KEY=)

you can pass -e NUMRECORDS=100 to change the number of records to extract from each crawl. 
you can pass -e NUMWARCS=1 to change the number of crawls to analyze. 

'''

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType
import sparknlp
spark = sparknlp.start() 
# sparknlp.start(gpu=True) >> for training on GPU
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from langdetect import detect
from pyspark.sql.functions import col, lit, concat_ws

from sklearn.metrics import classification_report
import requests
from warcio import ArchiveIterator
from bs4 import BeautifulSoup
import time
import pandas as pd
import re
import yfinance as yf
import boto3
import random
import sys 
import numpy as np
import os
#PARAMETERS
path_dl_model = './models/model_dl'
batch_size_max = sys.maxsize -1
num_records_percrawl = int(os.environ['NUMRECORDS']) #number of recors to attempt to extract from each crawl
ticker = 'SPY'
#read in financewordlist.csv into the list
wordlist = pd.read_csv('./sentdat/topics.csv', header=None)[0].tolist()
wordlist.extend(yf.Ticker(ticker).info['longName'].split())
number_warcs_to_analyze = int(os.environ['NUMWARCS']) #number of warcs to perform sentiment analysis on, goes from most reccent to farther back onse
randomsample = str(os.environ('RANSAMPLE')).lower() #Y or N, if Y, then it will take a random sample of warcs to analyze, if N, it will take the most recent warcs
#CREATING THE PIPELINE FOR LATER
document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(['sentence']) \
    .setOutputCol('token')

lemmatizer = Lemmatizer()\
    .setInputCols(['token'])\
    .setOutputCol('lemma')\
  .setDictionary("./sentdat/lemmas_small.txt", key_delimiter="->", value_delimiter="\t")
SentimentDetector = sentiment.SentimentDetector() \
    .setInputCols(['lemma', 'sentence'])\
    .setOutputCol('sentiment_score')\
    .setDictionary('./sentdat/sentiment-big.csv', ',')\

pipeline = Pipeline(stages=[
    document_assembler, 
    sentence_detector,
    tokenizer,
    lemmatizer,
    SentimentDetector
])
###GETTING WARC FILE NAMES FROM S3, GRABBING A RANDOM SAMPLE OF THEM
s3 = boto3.resource('s3')
my_bucket = s3.Bucket('commoncrawl')
warcs = []
for object in my_bucket.objects.filter(Prefix='crawl-data/CC-NEWS/'):
    if object.key.endswith('.warc.gz'):
        warcs.append(object.key)

if randomsample == 'y':
    warcs = random.sample(warcs, number_warcs_to_analyze)
else:
  warcs = warcs[-number_warcs_to_analyze:]

for index, warc in enumerate(warcs):
    warcs[index] = 'https://data.commoncrawl.org/' + warc

#function to convert time from commoncrawl format to y-m-d
def convert_header_date(date):
    return time.strftime('%Y-%m-%d', time.strptime(date, '%Y-%m-%dT%H:%M:%SZ'))


#obtaining stock data from yahoo finance from 2019 to current date.
currentdate = time.strftime("%Y-%m-%d")
stockdata = yf.download(ticker, start='2010-01-01', end=currentdate)['Adj Close']

#creating scehma to store text and prices
data = StructType([\
  StructField("text", StringType(), True),
    StructField("price", StringType(), True),
    StructField("date", StringType(), True)  
]
)
print('starting to load model')
FINDMODEL = PipelineModel.load(path_dl_model)
print('model loaded')

#function to drop non-finance articles
def drop_nonfinance_articles(df):
  df = FINDMODEL.transform(df)
  df = df.withColumn('finance', df['financial_model_pred.result'].getItem(0).cast('float'))
  df = df.filter(df['finance'] == 1.0)
  return df

batching_done = False
pausing_index = 0
while batching_done == False: 
  # creating the main rdd to store the data
  list_of_rows_batch = []
  rows_batch_len = 0
  recordsfetched = 0
  failures = 0
  datelist = []

  for index, warc_url in enumerate(warcs):
      response = requests.get(warc_url, stream=True)
      if response.ok!=True:
          raise Exception("Error downloading WARC file")
      records = ArchiveIterator(response.raw, arc2warc=True)
      #what this should do is write each record's plaintexxt to a csv file
      for record in records:
          if record.rec_type == 'response':
              try: 
                  html = record.content_stream().read() .decode('utf-8')
                  plaintext = BeautifulSoup(html, 'lxml').get_text()
                  plaintext = re.sub(r'\s+', ' ', plaintext)
                  plaintext = re.sub(r'[^a-zA-Z0-9\s]', '', plaintext).lower()

                  #obtains plaintext from the html
                  if detect(plaintext) == 'en' and len(plaintext) > 150:  
                      date = record.rec_headers.get_header('WARC-Date')
                      date = convert_header_date(date)
                      # append the plaintext and price to the batch
                      if date in stockdata.index:
                          datelist.append(date)
                          list_of_rows_batch.append({'text':plaintext, 'price':float(stockdata[date]), 'date':date})
                          recordsfetched += 1
                          rows_batch_len += 1
                      else:
                          print('date not in stockdata',date)
                          #likely a weekend or holiday, so we will just skip the entire warc
                          break
                  else:
                      recordsfetched += 1                          
              except:
                  recordsfetched += 1  # because if the entire warc file is not in english or wrong date, we still want to move on to the next one
                  failures += 1
                  #print("attempt record: ", record.rec_headers.get_header('WARC-Target-URI'), " failed")
                  pass

          if rows_batch_len >= batch_size_max:
              print("batch size max reached")
              pausing_index = index

          if recordsfetched >= num_records_percrawl:
              recordsfetched = 0
              print("warc done")
              break

      #finishing up for the last batch in it wasn't full and num batches wasnt maxed out.
  if rows_batch_len > 0:
      datelist = np.unique(datelist)
      newdatelist = []
      for date in datelist:
         newdatelist.append(str(date)) 
      datelist = newdatelist
      print("done one batch, size: ", rows_batch_len)
      rows_batch_len = 0

  if pausing_index != 0:
      warcs = warcs[pausing_index:]
      pausing_index = 0
  else:
      batching_done = True
  

  print("failures: ", failures)

  #now split list_of_rows_batch by dates in datelist
  
  list_of_lists_freach_date = []
  for date in datelist:
      list_of_lists_freach_date.append([row for row in list_of_rows_batch if row['date'] == date])
  
    
  sentscores = []
  finacial_data = []

  for index, one_date_lst in enumerate(list_of_lists_freach_date):
    df = spark.createDataFrame(one_date_lst, schema=data)
    #dropping non-finance articles
    df = drop_nonfinance_articles(df)
    numarticles = df.count() 
    print("dropped non-finance articles, num articles for this date: ", numarticles, " date: ", datelist[index])
    #drop unessecary columns created from dropping non-finance articles
    cols = df.columns
    for item in ['text', 'price', 'date']:
        cols.remove(item)
    df = df.drop(*cols)

    df = pipeline.fit(df).transform(df)

    df = df.withColumn("sentiment_score", concat_ws(",", "sentiment_score.result"))

    positives= df.filter(col('sentiment_score') == 'positive').count()
    negatives = numarticles - positives
    print("total positive and negatives for ", datelist[index])
    print("positives", positives)
    print("negatives", negatives)
    if negatives == 0:
      sentscores.append(positives)
    sentscores.append(positives/negatives)
    finacial_data.append(float(stockdata[datelist[index]]))



import matplotlib.pyplot as plt
x = np.arange(len(finacial_data))
plt.plot(x, finacial_data)
plt.xlabel('dates')
plt.ylabel('stock price')
plt.show()
plt.plot(x, sentscores)
plt.xlabel('dates')
plt.ylabel('sentiment score')
plt.show()
#normalize data 
finacial_data_plt = (finacial_data - np.mean(finacial_data))/np.std(finacial_data)
sentscores_plt = (sentscores - np.mean(sentscores))/np.std(sentscores)
plt.plot(x, finacial_data_plt)
plt.plot(x, sentscores_plt)
plt.xlabel('dates')
plt.ylabel('normalized stock price and sentiment score')
plt.show()

print(sentscores)
print(finacial_data)
print(datelist)