import sys,os,json,rdflib,re,copy,requests
from ptrcorrector import convex_hull


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

d = json.loads(open(sys.argv[1]).read()) #eg: model_folder_test31.1out.json
modelnum = sys.argv[2]
print(sys.argv[2])

ch = convex_hull.ConvexHull(modelnum)

qem = 0
qnem = 0
for idx,item in enumerate(d):
    print(item)
    print(str(item['uid']))
    print(item['question'])
    target = item['target'].split(' [sep]',1)[0]
    answer = item['answer_0'].split(' [sep]',1)[0]
    ents = item['goldents']
    rels = item['goldrels']
    print("ents: ",ents)
    print("rels: ",rels)
    print("target: ",target)
    print("answer: ",answer)
    wans_ = item['question'][0] + ' ' + answer
    if target.split() != answer.split():
        cmatch = False
#        print("query not match, try corrector with :",wans_)
#        for query in ch.correct(wans_):
#            print("alternatequery: ", query)
#            if query.split() == target.split():
#                print("target: ",target)
#                print("corque: ", query)
#                print("corrector matched")
#                qem += 1
#                cmatch = True
#                break
        if not cmatch:
            qnem += 1
            print("corrector fail")
    else:
        qem += 1
        print("querymatch")
 
    print('................')
    print("querymatch: ",qem," querynotmatch: ",qnem)

#f = open(sys.argv[3],'w')
#f.write(json.dumps(querywrong,indent=4,sort_keys=True))
#f.close()
