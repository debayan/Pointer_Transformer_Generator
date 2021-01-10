import tensorflow as tf
import sys
from utils import create_masks
from data_helper import Vocab
from fuzzywuzzy import fuzz


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
        for answer1stpass,target1stpass,uid,question1stpass,oov in zip(output1,labels["dec_target"],features["uid"],features["question"],features['question_oovs']):
            try:
                answerclean1stpass = answer1stpass[1:]
                target1stpass_ = ' '.join([vocab.id_to_word(x)  if x < vocab.size() else list(oov.numpy())[x - vocab.size()].decode('utf-8') for x in list(target1stpass.numpy()) if x != 1 and x != 3])
                answer1stpass_ = ' '.join([vocab.id_to_word(x)  if x < vocab.size() else list(oov.numpy())[x - vocab.size()].decode('utf-8') for x in list(answerclean1stpass.numpy()) if x != 1 and x!= 3])
                qcount1stpass += 1
                totalfuzz1stpass += fuzz.ratio(target1stpass_.lower(), answer1stpass_.lower())
                print("uid: ",int(uid.numpy()))
                print("1st pass question: ", question1stpass.numpy().decode('utf-8'))
                print("1st pass target: ", target1stpass_)
                print("1st pass answer: ", answer1stpass_,'\n')
            except Exception as err:
                print("err: ",err)
            
               
        print("1st pass avg fuzz after %d questions = %f"%(qcount1stpass,float(totalfuzz1stpass)/qcount1stpass))
    return output, attention_weights
