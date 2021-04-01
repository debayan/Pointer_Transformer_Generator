import tensorflow as tf
from utils import create_masks
import time
import sys
from data_helper import Data_Helper, Vocab
import time
from fuzzywuzzy import fuzz
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
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
        url = 'http://ltcpu1:8892/sparql/'
        #print(query)
        query = 'PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>  PREFIX dbo: <http://dbpedia.org/ontology/>  PREFIX res: <http://dbpedia.org/resource/> PREFIX dbp: <http://dbpedia.org/property/> ' + query
        r = requests.get(url, params={'format': 'json', 'query': query})
        json_format = r.json()
        #print(entid,json_format)
        results = json_format
        return results
    except Exception as err:
        print(err)
        return ''




def scce_with_ls(y, y_hat):
    y_hat = tf.one_hot(tf.cast(y_hat, tf.int32), y.shape[2]) #n_classes
    return categorical_crossentropy(y_hat, y, label_smoothing=0.2)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
                super(CustomSchedule, self).__init__()

                self.d_model = d_model
                self.d_model = tf.cast(self.d_model, tf.float32)

                self.warmup_steps = warmup_steps

        def __call__(self, step):
                arg1 = tf.math.rsqrt(step)
                arg2 = step * (self.warmup_steps ** -1.5)
                #print("learning rate: ", tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2))
                return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(loss_object, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = scce_with_ls(pred,real)
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)


def train_step(features, labels, params, model, optimizer, loss_object, train_loss_metric, batchcount, testbatcher, ckptstep):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(features["enc_input_mask"], labels["dec_input"])
        testlossfloat = 999.0
        with tf.GradientTape() as tape:
                questions = [q.numpy().decode('utf-8') for q in features["question"]]
                output, attn_weights = model(questions,features["enc_input"],features["extended_enc_input"], features["max_oov_len"], labels["dec_input"], training=params["training"], 
                                                                        enc_padding_mask=enc_padding_mask, 
                                                                        look_ahead_mask=combined_mask,
                                                                        dec_padding_mask=dec_padding_mask)
                loss = loss_function(loss_object, labels["dec_target"], output)
#                if batchcount%100 == 0 and batchcount > 1:
#                    vocab = Vocab(params['vocab_path'], params['vocab_size'])                                                  
#                    for answer,target,question in zip(output,labels["dec_target"],questions):
#                        target = ' '.join([vocab.id_to_word(x) for x in list(target.numpy()) if x != 1 and x != 3])
#                        print("trainquestion: ",question)
#                        print("traintarget: ", target)
#                        answer = ' '.join([vocab.id_to_word(x)  for x in list(tf.math.argmax(answer, axis=1).numpy()) if x != 1 and x!= 3])
#                        print("trainanswer: ",  answer,'\n')
        gradients = tape.gradient(loss, model.trainable_variables)        
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss_metric(loss)
        qcount = 0
        totalfuzz = 0.0
        totalfuzznonbeam = 0.0
        totf1 = 0
        avgf1 = 0.0
        if ckptstep%100 == 0 and ckptstep > 1:
            vocab = Vocab(params['vocab_path'], params['vocab_size'])
            
            for testidx,testbatch in enumerate(testbatcher):
                try:
                    output = tf.tile([[2]], [params["batch_size"], 1]) # 2 = start_decoding
                    testfeatures = testbatch[0]
                    testlabels = testbatch[1]
                    predblock = None
                    #testquestions = [q.numpy().decode('utf-8') for q in testfeatures["question"]]
                    for i in range(params["max_dec_len"]):
                        test_enc_padding_mask, test_combined_mask, test_dec_padding_mask = create_masks(testfeatures["enc_input_mask"], output)
                        predictions, test_attn_weights = model(testfeatures["question"],testfeatures["enc_input"],testfeatures["extended_enc_input"], testfeatures["max_oov_len"], output, training=False, enc_padding_mask=test_enc_padding_mask, look_ahead_mask=test_combined_mask,dec_padding_mask=test_dec_padding_mask)
                        # select the last word from the seq_len dimension
                        predblock = predictions
                        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
                        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
                        # concatentate the predicted_id to the output which is given to the decoder
                        # as its input.
                        output = tf.concat([output, predicted_id], axis=-1)
                        
                    idx = 0
                    for question,nonbeamanswer,target,uid,oov,ents,rels in zip(testfeatures["question"],output,testlabels["dec_target"],testfeatures["uid"],testfeatures['question_oovs'],testfeatures['ents'],testfeatures['rels']):
                        print("uid: ",int(uid.numpy()))
                        print("question: ", question.numpy().decode('utf-8'))
#                        answer = outputdict[idx]
                        idx+=1
                        words = []
                        for x in list(target.numpy()):
                            if x==3 or x==1:
                                break
                            if x < vocab.size():
                                words.append(vocab.id_to_word(x))
                            else:
                                words.append(list(oov.numpy())[x - vocab.size()].decode('utf-8'))
                        target_ = ' '.join(words)
                        words = []
                        prev = None
                        nonbeamans = list(nonbeamanswer[1:].numpy())
                        for idx,x in enumerate(nonbeamans):
                            if x==3 or x==1:
                                break

                            if x < vocab.size():
                                words.append(vocab.id_to_word(x))
                            else:
                                words.append(list(oov.numpy())[x - vocab.size()].decode('utf-8'))
                        nonbeamanswer_ = ' '.join(words)
                        answer_ = nonbeamanswer_
                        qcount += 1
                        totalfuzznonbeam += fuzz.ratio(target_.lower(), nonbeamanswer_.lower())
                        ents_ = [ent.decode('utf-8') for ent in ents.numpy()]
                        rels_ = [rel.decode('utf-8') for rel in rels.numpy()]
                        for idx1,ent in enumerate(ents_):
                            if ent:
                                target_ = target_.replace('entpos@@'+str(idx1+1),ent)
                        for idx1,rel in enumerate(rels_):
                            if rel:
                                target_ = target_.replace('predpos@@'+str(idx1+1),rel)
                        resulttarget = hitkg(target_)
                        for idx1,ent in enumerate(ents_):
                            if ent:
                                answer_ = answer_.replace('entpos@@'+str(idx1+1),ent)
                        for idx1,rel in enumerate(rels_):
                            if rel:
                                answer_ = answer_.replace('predpos@@'+str(idx1+1),rel)
                        resultanswer = hitkg(answer_)
                        f1  = calcf1(resulttarget,resultanswer)
                        totf1 += f1
                        avgf1 = totf1/float(qcount)
                        print("target: ", target_)
                        print("answer: ", answer_)
                        print("target: ",resulttarget)
                        print("answer: ",resultanswer)
                        print("f1: ",f1)
                        print("avgf1: ",avgf1)
                        print("nonbeam avg fuzz after %d questions = %f"%(qcount,float(totalfuzznonbeam)/qcount))
                    print("testidx: ",testidx)
                    #if testidx >= 4:
                    #    break
                except Exception as err:
                        print("er: ",err)             
        #return testlossfloat
#        return float(totalfuzznonbeam)/(qcount+0.001)
        return avgf1


def train_model(model, batcher, testbatcher, params, ckpt, ckpt_manager):
        learning_rate = CustomSchedule(params["model_depth"])
        #optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=0.1)  
        optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.98, epsilon=0.01)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
        train_loss_metric = tf.keras.metrics.Mean(name="train_loss_metric")

        try:
                bestf1 = 0.0
                epoch = 0
                while epoch < params["max_epochs"]:#while int(ckpt.step) < params["max_steps"]:
                        epoch += 1
                        for idx,batch in enumerate(batcher):
                                t0 = time.time()
                                valf1 = train_step(batch[0], batch[1], params, model, optimizer, loss_object, train_loss_metric, idx, testbatcher, ckpt.step)
                                t1 = time.time()
                                if ckpt.step%100 == 0 and ckpt.step > 1:
                                    print("valf1 - bestf1 : ",valf1,bestf1)
                                    if valf1 > bestf1:
                                        bestf1 = valf1
                                        print("Best val f1 so far: %f"%(valf1))
                                        ckpt_manager.save(checkpoint_number=int(ckpt.step))
                                        print("Saved checkpoint for step {}".format(int(ckpt.step)))
                                print("epoch {} step {}, time : {}, loss: {}".format(epoch,int(ckpt.step), t1-t0, train_loss_metric.result()))
                                ckpt.step.assign_add(1)
                        
        except KeyboardInterrupt:
                #ckpt_manager.save(int(ckpt.step))
                #print("Saved checkpoint for step {}".format(int(ckpt.step)))
                pass
        except Exception as err:
                print(err)
