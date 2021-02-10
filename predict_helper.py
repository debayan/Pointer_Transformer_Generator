import tensorflow as tf
import sys
from utils import create_masks
from data_helper import Vocab
from fuzzywuzzy import fuzz

def predict(featuress, params, model):
    vocab = Vocab(params['vocab_path'], params['vocab_size']) 
    totalfuzz = 0.0
    qcount = 0
    for features_ in featuress:
        features = features_[0]
        labels = features_[1]
        questions = [q.numpy().decode('utf-8') for q in features["question"]]
        output = tf.tile([[2]], [params["batch_size"], 1]) # 2 = start_decoding
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
        for answer,target,uid,question,oov in zip(output,labels["dec_target"],features["uid"],features["question"],features["question_oovs"]):
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
    
                prev = None
                nonbeamans = list(answerclean.numpy())
                words=[]
                for idx,x in enumerate(nonbeamans):
                    if x==3 or x==1:
                        break
                    if idx >= 3:#n-gram blocking where n = 2, dont let things like 'q123 p124 q123 p124' repeat
                        if nonbeamans[idx] == nonbeamans[idx-2] and nonbeamans[idx+1] == nonbeamans[idx-1]:
                            continue
                        if nonbeamans[idx] == nonbeamans[idx-2] and nonbeamans[idx-1] == nonbeamans[idx-3]:
                            continue
                    if x < vocab.size():
                        if vocab.id_to_word(x) == prev:
                            continue
                        words.append(vocab.id_to_word(x))
                        prev = vocab.id_to_word(x) #n-gram blocking where n = 1
                    else:
                        if list(oov.numpy())[x - vocab.size()].decode('utf-8') == prev: #n-gram blocking where n = 1
                            continue
                        if prev[0] == 'p' and list(oov.numpy())[x - vocab.size()].decode('utf-8')[0] == 'p': # dont let predicates repeat
                            continue
                        if prev[0] == 'q' and list(oov.numpy())[x - vocab.size()].decode('utf-8')[0] == 'q': # dont let entities repeat
                            continue
                        words.append(list(oov.numpy())[x - vocab.size()].decode('utf-8'))
                        prev = list(oov.numpy())[x - vocab.size()].decode('utf-8')
                answer_ = ' '.join(words)
                qcount += 1
                totalfuzz += fuzz.ratio(target_.lower(), answer_.lower())
                print("uid: ",int(uid.numpy()))
                print("question: ", question.numpy().decode('utf-8'))
                print("target: ", target_)
                print("answer: ", answer_,'\n')
            except Exception as err:
                print(err)
                continue
        print("avg fuzz after %d questions = %f"%(qcount,float(totalfuzz)/qcount))
    return output, attention_weights
