import re
import emoji
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


#case folding
def caseFold(tweet):
    tweet = tweet.lower() #lowercase
    tweet = re.sub("rt","", str(tweet)) #rt
    tweet = re.sub("user","",str(tweet)) # user
    tweet = re.sub("url","",str(tweet)) # url
    tweet = re.sub(r'[^\w\s]', '', str(tweet)) # punctuation
    tweet = re.sub(r"\d+", "", str(tweet)) # number
    
    # alpha numeric
    mod_string = ""
    for elem in tweet:
        if elem.isalnum() or elem == ' ':
            mod_string += elem
    tweet = mod_string
    
    #emoji
    res = emoji.emoji_list(tweet)
    if len(res) > 0 :
        tweet = emoji.replace_emoji(tweet, replace='')

    #whitespace
    tweet = " ".join(tweet.split())
    return tweet

# normalization (changes non formal words into the formal one)
def text_norm(text):
    reference = './repo_dataset/id-multi-label-hate-speech-and-abusive-language-detection/new_kamusalay.csv'
    df_ref = pd.read_csv(reference, encoding = "ISO-8859-1", header=None)
    tempText = text.split()
    
    for tempInd, tempVal in enumerate(tempText):
        for ind, item in enumerate(df_ref[0]):
            if tempVal == item:
                tempText[tempInd] = df_ref[1][ind]
    
    text = " ".join(tempText)
    return text



def stemming(text):

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    output = stemmer.stem(text)

    return output

def stopWordsRemoval(text):

    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    output = stopword.remove(text)
    
    return output

def cleanAll(text):
    text = caseFold(text)
    text = text_norm(text)
    text = stemming(text)
    text = stopWordsRemoval(text)
    
    return text