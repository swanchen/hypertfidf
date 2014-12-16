from collections import defaultdict
from collections import Counter
from nltk.corpus import wordnet
from gensim import corpora, models, similarities
from nltk.corpus import stopwords

import numpy
import httplib2
import json
import re
import nltk

stemmer_func = nltk.stem.snowball.EnglishStemmer().stem
access_token=''

def get_new_posts(access_token,id):
    all_data = []
    http = httplib2.Http(disable_ssl_certificate_validation=True)
    current_url = "https://graph.facebook.com/%(id)s/feed?access_token=%(access_token)s" %{"id":id,"access_token":access_token}
    while current_url!='':
        try:
            resp, raw = http.request(current_url, "GET")
            data = json.loads(raw)
            if data["data"]!='':
                all_data += data["data"]
                current_url=data["paging"]["next"]
        #print current_url
        except:
            #print "End of all posts"
            break
    filtered = []
    for entry in all_data:
        try:
            if entry["from"]["id"] == id:
                message = entry["message"] #.lower()
                filtered.append((message))
        except:
            print "Failed to parse: ",entry
    out=file('raw.txt','w')
    for i in range(len(filtered)):
        try:
            out.write(filtered[i])
            out.write('\n')
        except:
            continue
    return filtered

def get_words(filtered):
    p = re.compile("[^a-zA-Z]+")
    out=file('userpost.txt','w')
    for mesg in filtered:
        text = p.split(mesg)
        for word in text:
            out.write(word)
            out.write(' ')
    return out




