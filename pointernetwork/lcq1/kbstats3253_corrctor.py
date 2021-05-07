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
        
             
def hitkg(query,typeq):
    try:
        url = 'http://134.100.15.203:8892/sparql/'
        print(query)
        r = requests.get(url, params={'format': 'json', 'query': query})
        json_format = r.json()
        #print(entid,json_format)
        results = json_format
        if not results and typeq == 'target':
            print("no response")
            sys.exit(1)
        return results
    except Exception as err:
        print(err)
        if typeq == 'target':
            print("no response on target")
            sys.exit(1)
        return ''

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


def start(filename, modelnum):
    d = json.loads(open(filename).read()) #eg: model_folder_test31.1out.json
    ch = convex_hull.ConvexHull(modelnum)    
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
        resulttarget = hitkg(target,'target')
        resultanswer = hitkg(answer,'answer')
        wans_ = item['question'].split(' ')[0] + ' ' + ans_
        if empty(resultanswer):
            print("no answer")
            print("sending to corrector: ",wans_)
            for query in ch.correct(wans_):
                print("alternatequery: ", query)
                q = replace(query,ents,rels)
                print("replaces: ",q)
                try:
                    res = hitkg(q,'answer')
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
    #    print("original_quer: ",ques3253[item['uid']])
        print("gold: ",resulttarget)
        print("result: ",resultanswer)
        print('................')
        print("exactmatch: ",em, "  notmatch: ",nem," total: ",idx)
        print("querymatch: ",qem," querynotmatch: ",qnem)
        print("avg f1: ",avgf1)
    return avgf1

def f1eval(filename, modelnum):
    try:
        return start(filename, modelnum)
    except Exception as err:
        print(err)


if __name__ == '__main__':
    filename = sys.argv[1]
    modelnum = sys.argv[2]
    f1eval(filename, modelnum)
