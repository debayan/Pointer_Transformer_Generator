import sys
import json
import re
import requests
from elasticsearch import Elasticsearch
from textblob import TextBlob
import numpy as np
from multiprocessing import Pool
from fuzzywuzzy import fuzz

postags = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]

es = Elasticsearch(host="134.100.15.203",port=32816)


entembedcache = {}
labelembedcache = {}
relembedcache = {}

def getkgembedding(enturl):
    if enturl in entembedcache:
        return entembedcache[enturl]
    entityurl = '<http://www.wikidata.org/entity/'+enturl+'>'
    res = es.search(index="wikidataembedsindex01", body={"query":{"term":{"key":{"value":entityurl}}}})
    try:
        embedding = [float(x) for x in res['hits']['hits'][0]['_source']['embedding']]
        entembedcache[enturl] = embedding
        return embedding
    except Exception as e:
        print(enturl,' entity embedding not found')
        return 200*[0.0]
    return 200*[0.0]

def gettextmatchmetric(label,word):
    return [fuzz.ratio(label,word)/100.0,fuzz.partial_ratio(label,word)/100.0,fuzz.token_sort_ratio(label,word)/100.0]

def getlabelembedding(entid):
    if entid in labelembedcache:
        return labelembedcache[entid]
    res = es.search(index="wikidataentitylabelindex01", body={"query":{"term":{"uri":{"value":'http://wikidata.dbpedia.org/resource/'+entid}}}})
    if len(res['hits']['hits']) == 0:
        return [0]*300
    try:
        description = res['hits']['hits'][0]['_source']['wikidataLabel']
        r = requests.post("http://134.100.15.203:8887/ftwv",json={'chunks': [description]},headers={'Connection':'close'})
        labelembedding = r.json()[0]
        labelembedcache[entid] = labelembedding
        return labelembedding
    except Exception as e:
        print("getlabelembedding err: ",e)
        return [0.0]*300
    return [0.0]*300
    
def getrellabelembedding(rel,props):
    if rel in relembedcache:
        return relembedcache[rel]
    try:
        desc = props[rel]
        r = requests.post("http://134.100.15.203:8887/ftwv",json={'chunks': [desc]},headers={'Connection':'close'})
        descembedding = r.json()[0]
        relembedcache[rel] = descembedding
        return descembedding
    except Exception as e:
        print("getrellabelembedding err: ",e)
        return [0.0]*300
    return [0.0]*300
        
    


class Vectoriser():
    def __init__(self, proppath):
       print("Initialising Vectoriser")
       self.pool = Pool(4)
       self.props = {}
       for item in json.loads(open(proppath).read()):
           self.props[item['id']] = item['desc']
       print("Initialised Vectoriser")
   

    def vectorise(self, nlquery, sparql):
        if not nlquery:
            return []
        q = re.sub("\s*\?", "", nlquery.strip())
        ents = []
        rels = []
        ents = re.findall( r'wd:(.*?) ', sparql)
        rels = re.findall( r'wdt:(.*?) ',sparql)
        rels += re.findall( r'ps:(.*?) ',sparql)
        rels += re.findall( r'pq:(.*?) ',sparql)
        rels += re.findall( r'p:(.*?) ',sparql)
#        print("question: ",nlquery)
#        print("sparql: ", sparql)
#        print("entities: ",ents)
#        print("relations: ",rels)
        candidatetokens = []
        candidatevectors = []
        #questionembedding
        tokens = [token for token in q.split(" ") if token != ""]
        r = requests.post("http://134.100.15.203:8887/ftwv",json={'chunks': tokens},headers={'Connection':'close'})
        #print("r: ",r)
        questionembeddings = r.json()
        candidatevectors = [embedding+200*[0.0] for embedding in questionembeddings]#list(map(lambda x: sum(x)/len(x), zip(*questionembeddings)))
        candidatetokens += tokens
        candidatevectors.append(500*[-2.0]) #SEParator
        candidatetokens.append('[SEP]')
        for ent in ents:
            entityembedding = getkgembedding(ent)
            labelembedding = getlabelembedding(ent)
#            print("ent: ",ent)
#            print("embed: ",entityembedding)
#            print("labelembedding: ",labelembedding)
            candidatevectors.append(labelembedding+entityembedding) 
            candidatetokens.append(ent)
        candidatevectors.append(500*[-2.0]) #SEParator
        candidatetokens.append('[SEP]')
        for rel in rels:
            relembedding = getkgembedding(rel)
            labelembedding = getrellabelembedding(rel, self.props)
#            print("rel: ",rel)
#            print("embed: ",relembedding)
#            print("labelembedding: ",labelembedding)
            candidatevectors.append(labelembedding+relembedding)
            candidatetokens.append(rel)
        return candidatetokens,candidatevectors,ents,rels
        
        
        
if __name__ == '__main__':
    v = Vectoriser('wikidataprops.json')
#    print(v.vectorise("who is the president of India ?"))
    print(json.dumps(v.vectorise("What is Park Geun-hye real name, who wrote in Hanja?", "SELECT ?obj WHERE { wd:Q138048 p:P1559 ?s . ?s ps:P1559 ?obj . ?s pq:P282 wd:Q485619 }")))
