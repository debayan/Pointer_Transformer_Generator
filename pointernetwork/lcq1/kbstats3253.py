import sys,os,json,rdflib,re,copy,requests


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
        print(query)
        r = requests.get(url, params={'format': 'json', 'query': query})
        json_format = r.json()
        #print(entid,json_format)
        results = json_format
        return results
    except Exception as err:
        print(err)
        return ''

#load 3253
ques3253 = {}
f = open('Question-SPARQL_3253.csv')
for line in f.readlines():
    id = int(line.strip().split(',')[0][2:-1])
    print(id)
    sparql = ','.join(line.strip().split(',')[1:]).replace('"','')
    ques3253[id] = sparql
f.close()

goldd = {}
goldq = {}

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
    for idx1,ent in enumerate(ents):
        if ent:
            target = target.replace('entpos@@'+str(idx1+1),ent)
    for idx1,rel in enumerate(rels):
        if rel:
            target = target.replace('predpos@@'+str(idx1+1),rel)
    resulttarget = hitkg(target)
    for idx1,ent in enumerate(ents):
        if ent:
            answer = answer.replace('entpos@@'+str(idx1+1),ent)
    for idx1,rel in enumerate(rels):
        if rel:
            answer = answer.replace('predpos@@'+str(idx1+1),rel)
    resultanswer = hitkg(answer)
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
#    print("original_quer: ",ques3253[int(str(item['uid'])[1:])])
    print("gold: ",resulttarget)
    print("result: ",resultanswer)
    print('................')
    print("exactmatch: ",em, "  notmatch: ",nem," total: ",idx+1)
    print("querymatch: ",qem," querynotmatch: ",qnem)
    print("avg f1: ",avgf1)
    print("test split count: ", idx+1)
