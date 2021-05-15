import tensorflow as tf
import numpy as np
import ptrcorrector.pointer_net as pointer_net
import time
import os
import json
import sys

tf.app.flags.DEFINE_integer("batch_size1", 20,"Batch size.")
tf.app.flags.DEFINE_integer("max_input_sequence_len1", 200, "Maximum input sequence length.")
tf.app.flags.DEFINE_integer("max_output_sequence_len1", 200, "Maximum output sequence length.")
tf.app.flags.DEFINE_integer("rnn_size1", 32, "RNN unit size.")
tf.app.flags.DEFINE_integer("attention_size1", 50, "Attention size.")
tf.app.flags.DEFINE_integer("num_layers1", 1, "Number of layers.")
tf.app.flags.DEFINE_integer("beam_width1", 10, "Width of beam search .")
tf.app.flags.DEFINE_float("learning_rate1", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm1", 5.0, "Maximum gradient norm.")
tf.app.flags.DEFINE_boolean("forward_only1", False, "Forward Only.")
#tf.app.flags.DEFINE_string("log_dir", "./log", "Log directory")
tf.app.flags.DEFINE_string("data_path1", "./train.txt", "Data path.")
tf.app.flags.DEFINE_string("test_data_path1", "./test.txt", "Data path.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint1", 200, "frequence to do per checkpoint.")

FLAGS = tf.app.flags.FLAGS
#FLAGS.log_dir1='./'

class ConvexHull(object):
  def __init__(self, modelnum):
    self.modelnum = modelnum
    self.testgraph = tf.Graph()
    with self.testgraph.as_default():
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.operation_timeout_in_ms=6000
      self.testsess = tf.Session(config=config)
    self.build_model()
    #self.read_data()
    #self.test_read_data()
    self.id2word = json.loads(open('id2word_lcq13253_jivat_w_'+modelnum+'.txt').read())
    self.word2id = json.loads(open('word2id_lcq13253_jivat_w_'+modelnum+'.txt').read())
    

  def vectorise(self,tokenids):
    print("tokenids: ",tokenids)
    inputs = []
    enc_input_weights = []
    enc_input = []
    for t in tokenids:
      enc_input.append(int(t))
    enc_input_len = len(enc_input)
    enc_input += [0]*((FLAGS.max_input_sequence_len1-enc_input_len))
    enc_input = np.array(enc_input).reshape([-1,1])
    inputs.append(enc_input)
    weight = np.zeros(FLAGS.max_input_sequence_len1)
    weight[:enc_input_len]=1
    enc_input_weights.append(weight)
    self.test_inputs = np.stack(inputs)
    self.test_enc_input_weights = np.stack(enc_input_weights)
    print("Load inputs:            " +str(self.test_inputs.shape))
    print("Load enc_input_weights: " +str(self.test_enc_input_weights.shape))

  def build_model(self):
    with self.testgraph.as_default():
      self.testmodel = pointer_net.PointerNet(batch_size=1,
                    max_input_sequence_len=FLAGS.max_input_sequence_len1,
                    max_output_sequence_len=FLAGS.max_output_sequence_len1,
                    rnn_size=FLAGS.rnn_size1,
                    attention_size=FLAGS.attention_size1,
                    num_layers=FLAGS.num_layers1,
                    beam_width=FLAGS.beam_width1,
                    learning_rate=FLAGS.learning_rate1,
                    max_gradient_norm=FLAGS.max_gradient_norm1,
                    forward_only=True)

      # Prepare Summary writer
      # Try to get checkpoint
      print('./model'+self.modelnum)
      ckpt = tf.train.get_checkpoint_state('./model'+self.modelnum+'/solid/')
      print("Load model parameters from %s" % ckpt.model_checkpoint_path)
      self.testmodel.saver.restore(self.testsess, ckpt.model_checkpoint_path)


  def eval(self):
    """ Randomly get a batch of data and output predictions """  
    #inputs,enc_input_weights, outputs, dec_input_weights = self.get_test_batch()
    count = 0
    ans = None
    beams = []
    for input_,enc_input_weights_ in zip(self.test_inputs,self.test_enc_input_weights):
      try:
        predicted_ids = self.testmodel.step(self.testsess, [input_], [enc_input_weights_])
      except Exception as err:
        print(err)
        continue
      count += 1
      print("="*20)
      #print("inputs: ",inputs[i])
      #query = []
      #for x in input_:
      #  if x[0] == -1:
      #    break
      #  query.append(self.id2word[str(x[0])])   
      #print("input query: ", ' '.join(query))
      #print("* %dth sample target: %s" % (i,str(outputs[i,:]-2)))
      beams = []
      for predict in predicted_ids[0]:
        print("predict:",predict)
        predquery = []
        for x in predict:
          if x > 0:
            try:
                predquery.append(self.id2word[str(input_[x-1][0])])
            except Exception as err:
                print(err)
        ans = ' '.join(predquery)
        beams.append(ans)
        print("prediction: ", ans)
    return beams

  def correct(self,query):
    tokens = query.split(' ')
    try:
        tokenids = [self.word2id[x] for x in tokens]
    except Exception as err:
        print(err)
        return ''
    tokenids.append(-1)
    for x in self.id2word.keys():
      if int(x) in tokenids:
        continue
      else:
        tokenids.append(int(x))
    self.vectorise(tokenids)
    ans = self.eval()
    return ans

if __name__ == "__main__":
  c = ConvexHull()
  inputquery = 'select distinct ?vr0 where { <entpos@@1> <predpos@@1> ?vr0 }' 
  beams = c.correct(inputquery)
  print("input: ", inputquery)
  for ans in beams:
    print("corec: ",ans)
