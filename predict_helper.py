import tensorflow as tf
import sys
from utils import create_masks
from data_helper import Vocab
from fuzzywuzzy import fuzz
import requests


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
        url = 'https://aqqu.cs.uni-freiburg.de/sparql'
        query = 'PREFIX ns: <http://rdf.freebase.com/ns/>  ' + query
        print(query)
        r = requests.get(url, params={'format': 'json', 'query': query})
        json_format = r.json()
        print(json_format)
        results = json_format
        return results
    except Exception as err:
        print(err)
        return ''



def predict(featuress, params, model):
    vocab = Vocab(params['vocab_path'], params['vocab_size']) 
    totalfuzz = 0.0
    totf1 = 0
    qcount = 0
    em = 0
    retarr = []
    for features_ in featuress:
        features = features_[0]
        labels = features_[1]
        questions = [q.numpy().decode('utf-8') for q in features["question"]]
        output = tf.tile([[2]], [len(questions), 1]) # 2 = start_decoding
        for i in range(params["max_dec_len"]):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(features["enc_input_mask"], output)
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = model(questions,features["enc_input"],features["extended_enc_input"], features["max_oov_len"], output, training=False, 
                           enc_padding_mask=enc_padding_mask, 
                           look_ahead_mask=combined_mask,
                           dec_padding_mask=dec_padding_mask)
       
  
            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)
        for answer,target,uid,question,oov,ents,rels in zip(output,labels["dec_target"],features["uid"],features["question"],features["question_oovs"],features['ents'],features['rels']):
            resd = {}
            try:
                answerclean = answer[1:]
                words = []
                for x in list(target.numpy()):
                    if x==3 or x==1:
                        break
                    if x < vocab.size():
                        words.append(vocab.id_to_word(x))
                    else:
                        words.append(list(oov.numpy())[x - vocab.size()].decode('utf-8'))
                target_ = ' '.join(words)
    
                nonbeamans = list(answerclean.numpy())
                words=[]
                for idx,x in enumerate(nonbeamans):
                    if x==3 or x==1:
                        break
                    if x < vocab.size():
                        words.append(vocab.id_to_word(x))
                    else:
                        words.append(list(oov.numpy())[x - vocab.size()].decode('utf-8'))
                answer_ = ' '.join(words)
                qcount += 1
                ents_ = [ent.decode('utf-8') for ent in ents.numpy()]
                rels_ = [rel.decode('utf-8') for rel in rels.numpy()]
                totalfuzz += fuzz.ratio(target_.lower(), answer_.lower())
                if target_.lower() == answer_.lower():
                    em += 1
                targettemplate = target_
                answertemplate = answer_
                for idx1,ent in enumerate(ents_):
                    if ent:
                        target_ = target_.replace('entpos@@'+str(idx1+1),'ns:'+ent)
                for idx1,rel in enumerate(rels_):
                    if rel:
                        target_ = target_.replace('predpos@@'+str(idx1+1),'ns:'+rel)
                resulttarget = hitkg(target_)
                for idx1,ent in enumerate(ents_):
                    if ent:
                        answer_ = answer_.replace('entpos@@'+str(idx1+1),'ns:'+ent)
                for idx1,rel in enumerate(rels_):
                    if rel:
                        answer_ = answer_.replace('predpos@@'+str(idx1+1),'ns:'+rel)
                resultanswer = hitkg(answer_)
                f1  = calcf1(resulttarget,resultanswer)
                totf1 += f1
                avgf1 = totf1/float(qcount)
                print("uid: ",str(uid.numpy()))
                print("question: ", question.numpy().decode('utf-8'))
                print("target: ", target_)
                print("answer: ", answer_)
                print("targettemplate: ",targettemplate)
                print("answertemplate: ",answertemplate)
                print("goldents: ", ents_)
                print("goldrels: ", rels_)
                print("exactmatch: ",em)
                print("targetkg: ",resulttarget)
                print("answerkg: ",resultanswer)
                print("f1 = ",f1)
                print("avgf1 = ",avgf1)
                resd['uid'] = str(uid.numpy())
                resd['question'] = question.numpy().decode('utf-8')
                resd['target'] = target_
                resd['answer'] = answer_
                resd['targettemplate'] = targettemplate
                resd['answertemplate'] = answertemplate
                resd['goldents'] = ents_
                resd['goldrels'] = rels_
                retarr.append(resd) 
            except Exception as err:
                print(err)
                retarr.append(resd)
                continue
        print("avg fuzz after %d questions = %f"%(qcount,float(totalfuzz)/qcount))
    return output, attention_weights, retarr
