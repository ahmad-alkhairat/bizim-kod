'''
Created on 2017-12-29
@author: Ozan Tepe , Ahmad Alkhairat
'''
import os
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import csv
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import LdaModel
from properties import tweets_path
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from properties import username

tweets = []
docs = []


def read_tweets():
    
    # Read tweets from files
    if os.path.isdir(tweets_path):
        files = os.listdir(tweets_path)
        for filename in files:
            if username in filename:
                fullpath = os.path.join(tweets_path, filename)
                with open(fullpath, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        tweets.append(row['text'])
        print("Tweets loaded from file %s successfully..", username)
    else:
        print("Couldn't find file path..")


def preprocess_tweets():
    
    # Creating classes for cleaning process
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    en_stop = set(stopwords.words('english'))
    en_stop.add('http')
    p_stemmer = PorterStemmer()
    for tweet in tweets:
        
        # Tokenization
        tokens = tokenizer.tokenize(tweet.lower())   
        # print(tokens)
        print("Tokenization completed..")
        
        # Delete tokens with one length
        for t in tokens:
            if len(t) == 1:
                tokens.remove(t)
        
        # Stop words removal from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # print(stopped_tokens)
        print("Stop words removed from tokens..")    
        
        # Stemming
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        print("Stemming completed..")
        
        # Add cleaned tokens to list
        docs.append(stemmed_tokens)
        print("Cleaned tokens added to list..")
        
    print("Preprocessing completed..")


def implement_lda():
    
    # Turn our tokenized documents into a id <-> term dictionary
    dictionary = gensim.corpora.Dictionary(docs)
    dictionary.save('my_dictionary.dict')
    print("Dictionary saved..")
    
    # Convert tokenized documents into a document-term matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]
    gensim.corpora.MmCorpus.serialize('my_corpus.mm', doc_term_matrix)
    print("Corpus saved..")
    
    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel
    
    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=10, id2word=dictionary, passes=10)
    '''
    for i in ldamodel.print_topics(): 
        for j in i: 
            print(j)
    '''
    ldamodel.save('my_topic.model')
    print("LDA model saved..")
    
    

def show_results():
        
    # Load dictionary
    loaded_dict = gensim.corpora.Dictionary.load('my_dictionary.dict')
    
    # Load corpus
    loaded_corpus = gensim.corpora.MmCorpus('my_corpus.mm')
    
    # Load lda model
    loaded_model = LdaModel.load('my_topic.model')
    
    # Visualization of results
    vis_data = gensimvis.prepare(loaded_model, loaded_corpus, loaded_dict)
    pyLDAvis.show(vis_data)


if __name__ == '__main__':
    read_tweets()
    preprocess_tweets()
    implement_lda()
    show_results()

  
    

