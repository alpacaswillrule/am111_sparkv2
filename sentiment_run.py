'''
Usuage:
scp this file, a model called model_dl, and sentdat folder to the cluster, and the dockerfile
run the dockerfile with -e PYTHONFILETORUN=./sentiment_run.py
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
import botocore
import random
import sys 
#PARAMETERS
path_dl_model = '$(pwd)/models/model_dl'
numcrawlsforrun = 1
batch_size_max = sys.maxsize -1
num_records_percrawl = 150 #number of recors to attempt to extract from each crawl
ticker = 'SPY'
#read in financewordlist.csv into the list
wordlist = pd.read_csv('./sentdat/topics.csv', header=None)[0].tolist()
wordlist.extend(yf.Ticker(ticker).info['longName'].split())


###GETTING WARC FILE NAMES FROM S3, GRABBING A RANDOM SAMPLE OF THEM
s3 = boto3.resource('s3')
my_bucket = s3.Bucket('commoncrawl')
warcs = []
for object in my_bucket.objects.filter(Prefix='crawl-data/CC-NEWS/'):
    if object.key.endswith('.warc.gz'):
        warcs.append(object.key)

#choose 100 random warcs
randomwarcs = random.sample(warcs, numcrawlsforrun)

for index, warc in enumerate(randomwarcs):
    randomwarcs[index] = 'https://data.commoncrawl.org/' + warc

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
]
)


#function to drop non-finance articles
def drop_nonfinance_articles(df):
  model = PipelineModel.load(path_dl_model)
  df = model.transform(df)
  df = df.withColumn('finance', df['financial_model_pred.result'].getItem(0).cast('float'))
  df = df.filter(df['financial_model_pred.result'] == 1.0)
  return df



# creating the main rdd to store the data
df = spark.createDataFrame(spark.sparkContext.emptyRDD(), data)
list_of_rows_batch = []
rows_batch_len = 0
recordsfetched = 0
failures = 0

for warc_url in randomwarcs:
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
                        list_of_rows_batch.append({'text':plaintext, 'price':float(stockdata[date])})
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
            batchdf = spark.createDataFrame(list_of_rows_batch, data)
            print("union started")
            df = df.union(batchdf)
            print("union done")
            print(df.count())
            rows_batch_len = 0
            list_of_rows_batch = []
        if recordsfetched >= num_records_percrawl:
            recordsfetched = 0
            print("warc done")
            break

    #finishing up for the last batch in it wasn't full and num batches wasnt maxed out.
if rows_batch_len > 0:
    print(rows_batch_len)
    batchdf = spark.createDataFrame(list_of_rows_batch, data)
    df = df.union(batchdf)
    print("size of data: ", df.count())
    rows_batch_len = 0
print("done")
print("failures: ", failures)