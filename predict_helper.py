import tensorflow as tf
import sys
from utils import create_masks
from data_helper import Vocab
from fuzzywuzzy import fuzz
import numpy as np

def predict(featuress, params, model):
    vocab = Vocab(params['vocab_path'], params['vocab_size']) 
    totalfuzz = 0.0
    qcount = 0
    em = 0
    retarr = []
    
    for features_ in featuress:
        features = features_[0]
        labels = features_[1]
        questions = [q.numpy().decode('utf-8') for q in features["question"]]
        output = tf.tile([[2]], [len(questions), 1]) # 2 = start_decoding
        predblock = None
        for i in range(params["max_dec_len"]):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(features["enc_input_mask"], output)
#            print("encpaddingmask: ",enc_padding_mask)
#            print("cobinedmask: ",combined_mask)
#            print("decpaddingmask: ",dec_padding_mask)
#            print("featencinpmask: ",features["enc_input_mask"])
#            print("outpuit: ",output)
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = model(questions,features["enc_input"],features["extended_enc_input"], features["max_oov_len"], output, training=False, 
                           enc_padding_mask=enc_padding_mask, 
                           look_ahead_mask=combined_mask,
                           dec_padding_mask=dec_padding_mask)
       
  
            # select the last word from the seq_len dimension
            predblock = predictions
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)
        #beam search
        predblocktf = tf.transpose(predblock, [1, 0, 2])
        decoded, logprobs = tf.nn.ctc_beam_search_decoder(predblocktf, sequence_length=tf.constant(params["batch_size"]*[20]) ,beam_width=100,top_paths=10)
        beamdict = {}
        for x in range(10): #beam topk
            #print(x," ",decoded[x].indices.numpy())
            #print(x," ",decoded[x].values.numpy())
            indices = np.delete(decoded[x].indices.numpy(),1,axis=1)
            values = decoded[x].values.numpy()
            for idx,val in zip(indices,values):
                if idx[0] not in beamdict:
                    beamdict[idx[0]] = {}
                if x not in beamdict[idx[0]]:
                    beamdict[idx[0]][x] = []
                beamdict[idx[0]][x].append(val)
        print("beamdict: ",beamdict)

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
                
                beams = []
                for beamk in range(10):#beams topk
                    words = []
                    for idx,x in enumerate(beamdict[qcount][beamk]):
                        if x==3 or x==1:
                            break
                        if x < vocab.size():
                            words.append(vocab.id_to_word(x))
                        else:
                            words.append(list(oov.numpy())[x - vocab.size()].decode('utf-8'))
                    beamanswer = ' '.join(words)
                    beams.append(beamanswer)

                qcount += 1
                totalfuzz += fuzz.ratio(target_.lower(), answer_.lower())
                if target_.lower() == answer_.lower():
                    em += 1
                print("uid: ",int(uid.numpy()))
                print("question: ", question.numpy().decode('utf-8'))
                print("target: ", target_)
                print("answer: ", answer_)
                for beam in beams:
                    print("beam: ",beam)
                print("goldents: ",[ent.decode('utf-8') for ent in ents.numpy()])
                print("goldrels: ",[rel.decode('utf-8') for rel in rels.numpy()])
                print("exactmatch: ",em)
                resd['uid'] = int(uid.numpy())
                resd['question'] = question.numpy().decode('utf-8')
                resd['target'] = target_
                resd['answer'] = answer_
                resd['beams'] = beams
                resd['goldents'] = [ent.decode('utf-8') for ent in ents.numpy()]
                resd['goldrels'] = [rel.decode('utf-8') for rel in rels.numpy()]
                retarr.append(resd) 
            except Exception as err:
                print(err)
                retarr.append(resd)
                continue
        print("avg fuzz after %d questions = %f"%(qcount,float(totalfuzz)/qcount))
    return output, attention_weights, retarr
