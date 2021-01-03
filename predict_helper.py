import tensorflow as tf
import sys
from utils import create_masks
from data_helper import Vocab
from fuzzywuzzy import fuzz

def predict(featuress, params, model):
  output = tf.tile([[2]], [params["batch_size"], 1]) # 2 = start_decoding
  vocab = Vocab(params['vocab_path'], params['vocab_size']) 
  totalfuzz = 0.0
  qcount = 0
  for features_ in featuress:
    features = features_[0]
    labels = features_[1]
    questions = [q.numpy().decode('utf-8') for q in features["question"]]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(features["old_enc_input"], output)
    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = model(questions,features["old_enc_input"],features["extended_enc_input"], features["max_oov_len"], output, training=False, 
                             enc_padding_mask=enc_padding_mask, 
                             look_ahead_mask=combined_mask,
                             dec_padding_mask=dec_padding_mask)
   
    for answer,target,uid,question in zip(predictions,labels["dec_target"],features["uid"],features["question"]):
      print("answer: ", list(tf.math.argmax(answer, axis=1).numpy()))
      target = ' '.join([vocab.id_to_word(x) for x in list(target.numpy()) if x != 1 and x != 3])
      answer = ' '.join([vocab.id_to_word(x) for x in list(tf.math.argmax(answer, axis=1).numpy()) if x != 1 and x!= 3])
      qcount += 1
      totalfuzz += fuzz.ratio(target.lower(), answer.lower())
      print("uid: ",int(uid.numpy()))
      print("question: ", question.numpy().decode('utf-8'))
      print("target: ", target)
      print("answer: ", answer,'\n')
      print("avg fuzz after %d questions = %f"%(qcount,float(totalfuzz)/qcount))

    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return output, attention_weights
