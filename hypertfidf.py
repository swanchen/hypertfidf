# -*- coding: utf-8 -*-
import nltk
import math
from nltk.corpus import wordnet as wn
from collections import Counter
from nltk.corpus.reader import NOUN
#from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models, similarities
import inflect
import re
from operator import itemgetter, attrgetter #used for sort
from collections import defaultdict


wnl=WordNetLemmatizer()
p=inflect.engine()
dropwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now','day','days','week','weeks','month','months','year','years','second','seconds','minute','minutes','hour','hours','sunday','monday','tuesday','wednesday','thursday','friday','saturday','yesterday','today','tomorrow','tonight','night','weekday','weekend','weekends','time','times','moment','moments','am','pm','h','m','s','re','ll','d','isn','one','someone','somewhere','something','sometime','sometimes','anywhere','anyone','anytime','people','person','human','yes','no','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','ha','le','da']

def gethyper(filelist):
    HyperUserList=[]#all users' hypernyms
    for file in filelist:
        pnoun=[]
        word_list=[]
        temp=[]
        Hyperlist=[]
        file_word_morethanonce=[]
        
        open_file = open(file, 'r')
        contents=open_file.readlines()
        #only nouns
        for i in range(len(contents)):
            text=nltk.word_tokenize(contents[i].lower())
            token=nltk.pos_tag(text)
            word_list.extend(word for word,pos in token if pos=='NN' or pos=='NNS')
        #pure singular nouns
        for word in word_list:
            #get off stopwords
            #if word not in stopwords.words('english'):
            if word not in dropwords:
                try:
                    if word is not wnl.lemmatize(word,'n'):
                        word=p.singular_noun(word)
                    synsets=wn.synsets(word,NOUN)
                    if len(synsets)>0:
                        pnoun.append(word)
                except:
                    break
        #get hypernyms
        #print pnoun
        for i in range(len(pnoun)):
            try:
                w=wn.synsets(pnoun[i],wn.NOUN)
                for synset in w:
                    temp.extend(synset.hypernyms())
            except:
                print pnoun[i]
        for i in range(len(temp)):
            Hyperlist.extend(l.name for l in temp[i].lemmas)
        HyperUserList.append(Hyperlist)#cnt.most_common()[:len(cnt)]
    #delete hypernyms that appeared only once
    all_tokens = sum(HyperUserList, [])
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    file_word_morethanonce = [[word for word in text if word not in tokens_once]for text in HyperUserList]
    return file_word_morethanonce

def generate_dic(list):
    dictionary=corpora.Dictionary(list)#not include user
    dictionary.save('/tmp/user.dict')
    return dictionary

def user_vec(file,dictionary):
    word_list=[]
    pnoun=[]
    newfileword=[]
    temp=[]
    userHyper=[]
    word_morethanonce=[]
    
    open_file=open(file,'r')
    contents=open_file.readlines()
    for t in range(len(contents)):
        text=nltk.word_tokenize(contents[t].lower())
        token=nltk.pos_tag(text)
        #only nouns
        word_list.extend(word for word,pos in token if pos=='NN' or pos=='NNS')
    for word in word_list:
        try:
            if word is not wnl.lemmatize(word,'n'):
                word=p.singular_noun(word)
            synsets=wn.synsets(word,NOUN)
            if len(synsets)>0:
                pnoun.append(word)
        except:
            print word
    #extract stopwords for the first time
    for w in pnoun:
        #if w not in stopwords.words('english'):
        if w not in dropwords:#stopwords.words('english'):
            newfileword.append(w)
    print newfileword
    for i in range(len(newfileword)):#generate hypernyms
        try:
            w=wn.synsets(newfileword[i],wn.NOUN)
            for synset in w:
                temp.extend(synset.hypernyms())
        except:
            print newfileword[i]
    for i in range(len(temp)):
        userHyper.extend(l.name for l in temp[i].lemmas)
    #delete hypernyms that appeared only once
    d=defaultdict(int)
    for item in userHyper:
        d[item]+=1
    # remove words that appear only once
    tokens_once=[key for key,value in d.items() if value==1]
    word_morethanonce = [word for word in userHyper if word not in tokens_once]

    user_vec=dictionary.doc2bow(word_morethanonce)
    return user_vec

#corpus
def user_sims(list,vec,dictionary):
    corpus=[dictionary.doc2bow(text)for text in list]
    corpora.MmCorpus.serialize('/tmp/list.mm',corpus)
    tfidf=models.TfidfModel(corpus)
    #print tfidf[vec]
    #user similarity with firends
    index=similarities.SparseMatrixSimilarity(tfidf[corpus],num_features=len(dictionary.items()))
    sims=index[tfidf[vec]]
    print sims
