import sys,os,json,rdflib,re,copy,requests
from ptrcorrector import convex_hull 


def calcf1(target,answer):
    if not target:
        return 0.0
    if target == answer:
        return 1.0
    try:
        tb = target['results']['bindings']
        rb = answer['results']['bindings']
        tp = 0
        fp = 0
        fn = 0
        for r in rb:
            if r in tb:
                tp += 1
            else:
                fp += 1
        for t in tb:
            if t not in rb:
                fn += 1
        precision = tp/float(tp+fp+0.001)
        recall = tp/float(tp+fn+0.001)
        f1 = 2*(precision*recall)/(precision+recall+0.001)
        print("f1: ",f1)
        return f1
    except Exception as err:
        print(err)
    try:
        if target['boolean'] == answer['boolean']:
            print("boolean true/false match")
            f1 = 1.0
            print("f1: ",f1)
        if target['boolean'] != answer['boolean']:
            print("boolean true/false mismatch")
            f1 = 0.0
            print("f1: ",f1)
            return f1
    except Exception as err:
        f1 = 0.0
        print("f1: ",f1)
        return f1 
        
             
def hitkg(query):
    try:
        url = 'http://ltdocker:8894/sparql/'
        query = ''' PREFIX dbo: <http://dbpedia.org/ontology/>  
PREFIX res: <http://dbpedia.org/resource/>  
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>  
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>  
PREFIX dbp:  <http://dbpedia.org/property/>  ''' + query
        r = requests.get(url, params={'format': 'json', 'query': query})
        json_format = r.json()
        #print(entid,json_format)
        results = json_format
        return results
    except Exception as err:
        print(err)
        return ''

querywrong = []

modelnum = sys.argv[2]
print(sys.argv[2])

ch = convex_hull.ConvexHull(modelnum)
def replace(query,ents,rels):
    queryout = query
    for idx1,ent in enumerate(ents):
        if ent:
            queryout = queryout.replace('entpos@@'+str(idx1+1),ent)
    for idx1,rel in enumerate(rels):
        if rel:
            queryout = queryout.replace('predpos@@'+str(idx1+1),rel)
    return queryout

def empty(r):
    if 'boolean' not in r:
        if 'results' in r:
            if 'bindings' in r['results']:
                if not r['results']['bindings']:
                    return True
    return False



d = json.loads(open(sys.argv[1]).read()) #eg: model_folder_test31.1out.json

em = 0
nem = 0
qem = 0
qnem = 0
totf1 = 0.0
testqcount = 0
for idx,item in enumerate(d):
    print(item)
    print(str(item['uid']))
    print(item['question'])
    target = item['target']
    answer = item['answer']
    ents = item['goldents']
    rels = item['goldrels']
    print("ents: ",ents)
    print("rels: ",rels)
    print("target: ",target)
    print("answer: ",answer)
    if target == answer:
        qem += 1
        print("querymatch")
    else:
        qnem += 1
        print("querynotmatch")
    targ_ = item['querytemptar']
    ans_ = item['querytempans']
    target = replace(target,ents,rels)
    print("replaced: ",target)
    answer = replace(answer,ents,rels)
    print("replaced: ",answer)
    resulttarget = hitkg(target)
    resultanswer = hitkg(answer)
    wans_ = item['question'].split(' ')[0] + ' ' + ans_
    if empty(resultanswer):
        print("no answer")
        print("sending to corrector: ",wans_)
        for query in ch.correct(wans_):
            print("alternatequery: ", query)
            q = replace(query,ents,rels)
            print("replaces: ",q)
            try:
                res = hitkg(q)
                print("result: ",res)
            except Exception as err:
                print(err)
                continue
            if not empty(res):
                resultanswer = res
                break
    if not empty(resulttarget):
        f1  = calcf1(resulttarget,resultanswer)
    else:
        f1 = 0.0
    totf1 += f1
    avgf1 = totf1/float(idx+1)
    if resulttarget == resultanswer:
        print("match")
        em += 1
    else:
        print("nomatch")
        nem += 1
        querywrong.append({'querytempans':ans_, 'querytemptar': targ_, 'queryans':answer,'querytar':target,'id':str(item['uid']),'question':item['question'],'ents':ents,'rels':rels,'resulttarget':resulttarget,'resultanswer':resultanswer})
    print("target_filled: ",target)
    print("answer_filled: ",answer)
#    print("original_quer: ",ques3253[item['uid']])
    print("gold: ",resulttarget)
    print("result: ",resultanswer)
    print('................')
    print("exactmatch: ",em, "  notmatch: ",nem," total: ",idx)
    print("querymatch: ",qem," querynotmatch: ",qnem)
    print("avg f1: ",avgf1)


