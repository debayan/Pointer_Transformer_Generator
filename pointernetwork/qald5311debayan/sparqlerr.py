import sys,os,json,rdflib,re,copy,requests



cq = 0
def hitkg(query):
    try:
        url = 'http://ltdocker:8894/sparql/'
        query = 'PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>  PREFIX dbo: <http://dbpedia.org/ontology/>  PREFIX res: <http://dbpedia.org/resource/> PREFIX dbp: <http://dbpedia.org/property/> PREFIX yago: <http://dbpedia.org/class/yago/> ' + query
        #print(query)
        r = requests.get(url, params={'format': 'json', 'query': query})
        json_format = r.json()
        results = json_format
        return results
    except Exception as err:
        print(query)
        print("err: ",err)
        global cq
        cq += 1
        return ''

d = json.loads(open(sys.argv[1]).read()) #eg: model_folder_test31.1out.json

querywrong = []

em = 0
nem = 0
qem = 0
qnem = 0
totf1 = 0.0
for idx,item in enumerate(d):
#    print(item)
#    print(str(item['uid']))
#    print(item['question'])
    answer = item['answer']
    ents = item['goldents']
    rels = item['goldrels']
#    print("ents: ",ents)
#    print("rels: ",rels)
#    print("answer: ",answer)
    for idx1,ent in enumerate(ents):
        if ent:
            answer = answer.replace('entpos@@'+str(idx1+1),ent)
    for idx1,rel in enumerate(rels):
        if rel:
            answer = answer.replace('predpos@@'+str(idx1+1),rel)
    resultanswer = hitkg(answer)

print("wrong: ",cq," total: ",idx+1)
