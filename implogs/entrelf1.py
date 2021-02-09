import sys,os,json


def extractpredentrels(line):
    predents = set()
    predrels = set()
    entsrels = line.split('[sep]')[1:]
    for phrase in entsrels:
        for word in phrase.split(' '):
            if word:
                if word[0] == 'p':
                    predrels.add(word)
                if word[0] == 'q':
                    predents.add(word)
    return predents,predrels

f = open(sys.argv[1])
lines = f.readlines()
tpe = 0
fpe = 0
fne = 0
tpr = 0
fpr = 0
fnr = 0
for idx,line in enumerate(lines):
	if 'uid:' in line:
		print(line) #uid
		print(lines[idx+1]) #question
		print(lines[idx+2]) #target
		print(lines[idx+3])
		try:
			question,ents,rels = lines[idx+2].strip().split('[sep]')
		except Exception as err:
			print(err)
			continue
		goldents = set()
		goldrels = set()
		[goldents.add(x) for x in ents.split(' ') if x]
		[goldrels.add(x) for x in rels.split(' ') if x]
		#print(ents,rels,goldents,goldrels)
		predents, predrels = extractpredentrels(lines[idx+3].strip())
		print("goldents: ",goldents)
		print("goldrels: ",goldrels)
		print("predents: ",predents)
		print("predrels: ",predrels)
		#compute entity f1
		for goldentity in goldents:
			if goldentity in predents:
				tpe += 1
			else:
				fne += 1
		for queryentity in predents:
			if queryentity not in goldents:
				fpe += 1
		precisione = tpe/float(tpe+fpe+0.001)
		recalle = tpe/float(tpe+fne+0.001)
		f1e = 2*(precisione*recalle)/(precisione+recalle+0.001) 
		print("precisione: %f recalle: %f f1e: %f"%(precisione, recalle, f1e))
		#compute relation f1
		for goldrel in goldrels:
			if goldrel in predrels:
				tpr += 1
			else:
				fnr += 1
		for queryrel in predrels:
			if queryrel not in goldrels:
				fpr += 1
		precisionr = tpr/float(tpr+fpr+0.001)
		recallr = tpr/float(tpr+fnr+0.001)
		f1r = 2*(precisionr*recallr)/(precisionr+recallr+0.001)
		print("precisionr: %f recallr: %f f1r: %f"%(precisionr, recallr, f1r))
