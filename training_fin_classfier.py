  
'''
  USUAGE: 
  note you have to use ec2-user as username when ssh and scp
  CREATE CLUSTER WITH SPARK-DEV.IPYNB
  SCP THIS FILE, SENTDAT, ARTICLESPAR,DOCKERFILE TO THE CLUSTER
  THEN SSH ONTO CLUSTER ENTER THE DIRECTORY WITH THE FILES
  THEN BUILD DOCKERFILE AND RUN THE CONTAINER WITH
  sudo docker build -t <name> .
  AND RUN WITH
  sudo docker run -v "$(pwd)/models:/models" your_image_name
  to save model to models folder on hostmachine
  EXAMPLE SCP AND SSH COMMANDS EXIST AT THE BEGINNING OF SPARK-DEV FILE.
'''

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType
import sparknlp
spark = sparknlp.start()\
    # .config("spark.executorEnv.YARN_CONTAINER_RUNTIME_TYPE", "docker")\
    # .config("spark.executorEnv.YARN_CONTAINER_RUNTIME_DOCKER_IMAGE", "TODO")\
    #     .config("spark.appMasterEnv.YARN_CONTAINER_RUNTIME_TYPE", "docker")\
    # .config("spark.appMasterEnv.YARN_CONTAINER_RUNTIME_DOCKER_IMAGE", "TODO")\
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
import math
import os
#PARAMETERS

numcrawlsforrun = 5
batch_size_max = sys.maxsize -1
num_records_percrawl = 120 #number of recors to attempt to extract from each crawl, or takes all the records if less.
ticker = 'SPY'
ratio_com_yfin = .7 #for every commoncrawl article, how many yahoo finance articles to include
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

def contains_stock(plaintext, stklist=wordlist): 
    try:
        for stk in stklist:
            if plaintext.find(stk) != -1:
                return True
        return False
    except:
        raise Exception("issue with wordlist")

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
                if detect(plaintext) == 'en' and len(plaintext) > 150 and contains_stock(plaintext) == False:  #TODO add classifier here later to ensure its a financial article
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
    rows_batch_len = 0
print("done")
print("failures: ", failures)

#print all files in the directory
print(os.listdir('./'))

articles = spark.read.parquet(f'./articlespar.parquet').limit(math.floor(df.count()*ratio_com_yfin))

df = df.withColumn('financial', lit(0))
df = df.union(articles)

#split the data into training and testing
train, test = df.randomSplit([0.8, 0.2], seed=3204123)


# preprocess the text data
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")

normalizer = Normalizer()\
    .setInputCols(["token"])\
    .setOutputCol("normalized")\
    .setLowercase(True)

stopwords_cleaner = StopWordsCleaner()\
    .setInputCols("normalized")\
    .setOutputCol("cleanTokens")\
    .setCaseSensitive(False)

lemma = LemmatizerModel.pretrained("lemma_antbnc")\
    .setInputCols(["cleanTokens"])\
    .setOutputCol("lemma")

word_embeddings = BertEmbeddings\
    .pretrained('bert_base_cased', 'en') \
    .setInputCols(["document",'lemma'])\
    .setOutputCol("embeddings")\

# https://nlp.johnsnowlabs.com/docs/en/transformers#bertsentenceembeddings? better for sentence embeddings for later models, this one words is better
#https://nlp.johnsnowlabs.com/docs/en/transformers#debertaembeddings
#lots of transfoemrs to choose from for later tasks, for this one lightweight bert might be the best
embeddingsSentence = SentenceEmbeddings()\
    .setInputCols(["document", "embeddings"])\
    .setOutputCol("sentence_embeddings")\
    .setPoolingStrategy("AVERAGE")

classifierdl = ClassifierDLApproach()\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("financial_model_pred")\
    .setLabelColumn("financial")\
    .setMaxEpochs(5)\
        .setEnableOutputLogs(True)\
    .setLr(0.001)\

DLpipeline = Pipeline(
    stages = [
        document_assembler,
        tokenizer,
        normalizer,
        stopwords_cleaner,
        lemma,
        word_embeddings,
        embeddingsSentence,
        classifierdl
    ])



DLpipelineModel = DLpipeline.fit(train)
timevar = time.time()
DLpipelineModel.save(f"./models/dl_model_{numcrawlsforrun*num_records_percrawl}_{ratio_com_yfin}_{timevar}")


# model = PipelineModel.load("dl_model")
print("done training and saving model")
test_predict = DLpipelineModel.transform(test)



results = test_predict.select('text','price', 'financial','financial_model_pred.result')
results = results.withColumn('result', results['result'].getItem(0).cast('float'))

results = results.withColumn('result', results['result'].cast('float'))
print("done predicting, here are results on the test set")
print(classification_report(results.select('financial').collect(), results.select('result').collect()), 'green')

#https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT.ipynb
#https://towardsdatascience.com/text-classification-in-spark-nlp-with-bert-and-universal-sentence-encoders-e644d618ca32
#can get bert from there, then create a piepline that uses the bert model to get embeddings, then use the embeddings to train a classifier
#then we conver this to a script, upload to emr and get a large scale model.