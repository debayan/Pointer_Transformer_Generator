import sys,os,json,rdflib,re,copy,requests
from ptrcorrector.convex_hull import ConvexHull


def calcf1(target,answer):
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
        url = 'http://localhost:8892/sparql/'
        #print(query)
        r = requests.get(url, params={'format': 'json', 'query': query})
        json_format = r.json()
        #print(entid,json_format)
        results = json_format
        return results
    except Exception as err:
        print(err)
        return ''

goldd = {}
goldq = {}
d = json.loads(open(sys.argv[1]).read()) # test-data.json  (lcq1 test file)

for item in d:
    result = hitkg(item['sparql_query'])
#    print(item)
#    print(result)
    goldd[str(item['_id'])] = result
    goldq[str(item['_id'])] = item['sparql_query']

d = json.loads(open(sys.argv[2]).read()) #eg: model_folder_test31.1out.json

querywrong = []

ch = ConvexHull()
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
       

em = 0
nem = 0
qem = 0
qnem = 0
totf1 = 0.0
for idx,item in enumerate(d):
    print(item)
    print(item['question'])
    target = item['target'].split('[sep]')[0].strip()
    answer = item['answer_0'].split('[sep]')[0].strip()
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
    targ_ = target
    ans_ = answer
    target = replace(target,ents,rels)
    print("replaced: ",target)
    answer = replace(answer,ents,rels)
    print("replaced: ",answer)
    resulttarget = hitkg(target)
    resultanswer = hitkg(answer)
    if empty(resultanswer):
        print("no answer")
        for query in ch.correct(ans_):
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
    f1  = calcf1(resulttarget,resultanswer)
    totf1 += f1
    avgf1 = totf1/float(idx+1)
    if resulttarget == resultanswer:
        print("match")
        em += 1
    else:
        print("nomatch")
        nem += 1
    print("target_filled: ",target)
    print("answer_filled: ",answer)
    print("gold: ",resulttarget)
    print("result: ",resultanswer)
    print('................')
    print("exactmatch: ",em, "  notmatch: ",nem," total: ",idx)
    print("querymatch: ",qem," querynotmatch: ",qnem)
    print("avg f1: ",avgf1)

#f = open(sys.argv[3],'w')
#f.write(json.dumps(querywrong,indent=4,sort_keys=True))
#f.close()