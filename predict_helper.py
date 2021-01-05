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
        output = tf.tile([[2]], [params["batch_size"], 1]) # 2 = start_decoding
        
        for i in range(params["max_dec_len"]):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(features["old_enc_input"], output)
            predictions, attention_weights = model(questions,features["old_enc_input"],features["extended_enc_input"], features["max_oov_len"], output, training=False, 
                             enc_padding_mask=enc_padding_mask, 
                             look_ahead_mask=combined_mask,
                             dec_padding_mask=dec_padding_mask)
            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)
        questions2ndpass = []
        for answer,target,uid,question in zip(output,labels["dec_target"],features["uid"],features["question"]):
            answerclean = answer[1:]
            target_ = ' '.join([vocab.id_to_word(x) for x in list(target.numpy()) if x != 1 and x != 3])
            answer_ = ' '.join([vocab.id_to_word(x) for x in list(answerclean.numpy()) if x != 1 and x!= 3])
            qcount1stpass += 1
            totalfuzz1stpass += fuzz.ratio(target_.lower(), answer_.lower())
            questions2ndpass.append(question+' @@END@@ '+answer_)
            print("uid: ",int(uid.numpy()))
            print("1st pass question: ", question.numpy().decode('utf-8'))
            print("1st pass target: ", target_)
            print("1st pass answer: ", answer_,'\n')
            
        testbatch = testbatcher(questions2ndpass,params["vocab_path"],params) 
        featuress = testbatch
        questions = [q.numpy().decode('utf-8') for features in featuress for q in features["question"]]
        output = tf.tile([[2]], [params["batch_size"], 1]) # 2 = start_decoding
        for i in range(params["max_dec_len"]):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(features["old_enc_input"], output)
            predictions, attention_weights = model(questions,features["old_enc_input"],features["extended_enc_input"], features["max_oov_len"], output, training=False,
                             enc_padding_mask=enc_padding_mask,
                             look_ahead_mask=combined_mask,
                             dec_padding_mask=dec_padding_mask)
            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)
        for answer,target,uid,question in zip(output,labels["dec_target"],features["uid"],features["question"]):
            answerclean = answer[1:]
            target_ = ' '.join([vocab.id_to_word(x) for x in list(target.numpy()) if x != 1 and x != 3])
            answer_ = ' '.join([vocab.id_to_word(x) for x in list(answerclean.numpy()) if x != 1 and x!= 3])
            totalfuzz2ndpass += fuzz.ratio(target_.lower(), answer_.lower())
            qcount2ndpass += 1
            print("uid: ",int(uid.numpy()))
            print("2nd pass question: ", question.numpy().decode('utf-8'))
            print("2nd pass target: ", target_)
            print("2nd pass answer: ", answer_,'\n')
            print("1st pass avg fuzz after %d questions = %f"%(qcount1stpass,float(totalfuzz1stpass)/qcount1stpass))
            print("2nd pass avg fuzz after %d questions = %f"%(qcount2ndpass,float(totalfuzz2ndpass)/qcount2ndpass))
    return output, attention_weights
