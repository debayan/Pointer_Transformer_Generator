import sys,os,json

word2id = {}
id2word = {}
d = json.loads(open(sys.argv[1]).read())
wcount = 3
for item in d:
    if 'UNK' in item['querytempans'] or 'UNK' in item['querytemptar']:
        continue
    if item['querytempans'] == item['querytemptar']:
        continue
    for w in item['querytempans'].split():
        if w not in word2id:
            word2id[w] = wcount
            id2word[wcount] = w
            wcount += 1
    for w in item['querytemptar'].split():
        if w not in word2id:
            word2id[w] = wcount
            id2word[wcount] = w
            wcount += 1
master = []
for item in d:
    if 'UNK' in item['querytempans'] or 'UNK' in item['querytemptar']:
        continue
    if item['querytempans'] == item['querytemptar']:
        continue
    arr = []
    for w in item['querytempans'].split():
        arr.append(str(word2id[w]))
    arr.append(str(-1))
    for x in id2word.keys():
        if (str(x)) in arr:
            continue
        else:
            arr.append(str(x))
    arr.append('output')
    oarr = []
    for w in item['querytemptar'].split():
        idx = arr.index(str(word2id[w]))
        oarr.append(str(idx+1))
    arr += oarr
    print(arr)
    print(item['querytempans'])
    print(item['querytemptar'])
    master.append(arr)

f = open('input_qald5_newvocab_jivat.txt','w')
for x in master:
    f.write(' '.join(x)+'\n')
f.close()

f = open('id2word_qald5_newvocab_jivat.txt','w')
f.write(json.dumps(id2word))
f.close()

f = open('word2id_qald5_newvocab_jivat.txt','w')
f.write(json.dumps(word2id))
f.close()
