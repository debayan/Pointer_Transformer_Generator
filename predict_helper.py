import tensorflow as tf
import sys
from utils import create_masks
from data_helper import Vocab
from fuzzywuzzy import fuzz
from entitybatcher import testbatcher


def predict(featuress, params, model):
    vocab = Vocab(params['vocab_path'], params['vocab_size']) 
    totalfuzz1stpass = 0.0
    totalfuzz2ndpass = 0.0
    qcount1stpass = 0
    qcount2ndpass = 0
    for features_ in featuress:
        features = features_[0]
        labels = features_[1]
        questions = [q.numpy().decode('utf-8') for q in features["question"]]
        output1 = tf.tile([[2]], [params["batch_size"], 1]) # 2 = start_decoding
        for i in range(params["max_dec_len"]):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(features["old_enc_input"], output1)
            predictions1, attention_weights = model(questions,features["old_enc_input"],features["extended_enc_input"], features["max_oov_len"], output1, training=False, 
                             enc_padding_mask=enc_padding_mask, 
                             look_ahead_mask=combined_mask,
                             dec_padding_mask=dec_padding_mask)
            # select the last word from the seq_len dimension
            predictions1 = predictions1[: ,-1:, :]  # (batch_size, 1, vocab_size)
            predicted_id1 = tf.cast(tf.argmax(predictions1, axis=-1), tf.int32)
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output1 = tf.concat([output1, predicted_id1], axis=-1)
        questions2ndpass = []
        for answer1stpass,target1stpass,uid,question1stpass in zip(output1,labels["dec_target"],features["uid"],features["question"]):
            answerclean1stpass = answer1stpass[1:]
#            target1stpass_ = ' '.join([vocab.id_to_word(x) for x in list(target1stpass.numpy()) if x != 1 and x != 3])
            answer1stpass_ = ' '.join([vocab.id_to_word(x) for x in list(answerclean1stpass.numpy()) if x != 1 and x!= 3])
#            qcount1stpass += 1
#            totalfuzz1stpass += fuzz.ratio(target1stpass_.lower(), answer1stpass_.lower())
            questions2ndpass.append(question1stpass.numpy().decode('utf-8')+' @@END@@ '+answer1stpass_)
#            print("uid: ",int(uid.numpy()))
#            print("1st pass question: ", question1stpass.numpy().decode('utf-8'))
#            print("1st pass target: ", target1stpass_)
#            print("1st pass answer: ", answer1stpass_,'\n')
            
        testbatch = testbatcher(questions2ndpass,params["vocab_path"],params)
        testfeaturess = testbatch
        for testfeatures in testfeaturess:
            questions = [q.numpy().decode('utf-8') for q in testfeatures["question"]]
            output2 = tf.tile([[2]], [params["batch_size"], 1]) # 2 = start_decoding
            for i in range(params["max_dec_len"]):
                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(testfeatures["old_enc_input"], output2)
                predictions2, attention_weights = model(questions,testfeatures["old_enc_input"],testfeatures["extended_enc_input"], testfeatures["max_oov_len"], output2, training=False,
                             enc_padding_mask=enc_padding_mask,
                             look_ahead_mask=combined_mask,
                             dec_padding_mask=dec_padding_mask)
                # select the last word from the seq_len dimension
                predictions2 = predictions2[: ,-1:, :]  # (batch_size, 1, vocab_size)
                predicted_id2 = tf.cast(tf.argmax(predictions2, axis=-1), tf.int32)
                # concatentate the predicted_id to the output which is given to the decoder
                # as its input.
                output2 = tf.concat([output2, predicted_id2], axis=-1)
            for answer1stpass,answer2ndpass,target2ndpass,uid,question1stpass,question2ndpass in zip(output1,output2,labels["dec_target"],testfeatures["uid"],features["question"],questions):
                answerclean1stpass = answer1stpass[1:]
                target1stpass_ = ' '.join([vocab.id_to_word(x) for x in list(target1stpass.numpy()) if x != 1 and x != 3])
                answer1stpass_ = ' '.join([vocab.id_to_word(x) for x in list(answerclean1stpass.numpy()) if x != 1 and x!= 3])
                qcount1stpass += 1
                totalfuzz1stpass += fuzz.ratio(target1stpass_.lower(), answer1stpass_.lower())
                
                questions2ndpass.append(question1stpass.numpy().decode('utf-8')+' @@END@@ '+answer1stpass_)
                print("uid: ",int(uid.numpy()))
                print("1st pass question: ", question1stpass.numpy().decode('utf-8'))
                answerclean2 = answer2ndpass[1:]
                target2ndpass_ = ' '.join([vocab.id_to_word(x) for x in list(target2ndpass.numpy()) if x != 1 and x != 3])
                answer2ndpass_ = ' '.join([vocab.id_to_word(x) for x in list(answerclean2.numpy()) if x != 1 and x!= 3])
                totalfuzz2ndpass += fuzz.ratio(target2ndpass_.lower(), answer2ndpass_.lower())
                qcount2ndpass += 1
                print("2nd pass question: ", question2ndpass)
                print("1st pass target: ", target1stpass_)
                print("1st pass answer: ", answer1stpass_)
                print("2nd pass answer: ", answer2ndpass_,'\n')
            print("1st pass avg fuzz after %d questions = %f"%(qcount1stpass,float(totalfuzz1stpass)/qcount1stpass))
            print("2nd pass avg fuzz after %d questions = %f"%(qcount2ndpass,float(totalfuzz2ndpass)/qcount2ndpass))
                
    return output, attention_weights
