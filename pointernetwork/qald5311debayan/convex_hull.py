import tensorflow as tf
import numpy as np
import pointer_net 
import time
import os
import json
import sys
import random
from fuzzywuzzy import fuzz

tf.app.flags.DEFINE_integer("batch_size", 10,"Batch size.")
tf.app.flags.DEFINE_integer("max_input_sequence_len", 200, "Maximum input sequence length.")
tf.app.flags.DEFINE_integer("max_output_sequence_len", 200, "Maximum output sequence length.")
tf.app.flags.DEFINE_integer("rnn_size", 32, "RNN unit size.")
tf.app.flags.DEFINE_integer("attention_size", 100, "Attention size.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers.")
tf.app.flags.DEFINE_integer("beam_width", 5, "Width of beam search .")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Maximum gradient norm.")
tf.app.flags.DEFINE_boolean("forward_only", False, "Forward Only.")
#tf.app.flags.DEFINE_string("log_dir", "./log", "Log directory")
tf.app.flags.DEFINE_string("data_path", "./traindebqald5.txt", "Data path.")
tf.app.flags.DEFINE_string("test_data_path", "./testnewvocabjivat.txt", "Data path.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "frequence to do per checkpoint.")
tf.app.flags.DEFINE_string("modelnum", "0", "model number")

FLAGS = tf.app.flags.FLAGS
FLAGS.log_dir='./model'+FLAGS.modelnum

class ConvexHull(object):
  def __init__(self, forward_only):
    self.forward_only = forward_only
    self.graph = tf.Graph()
    self.testgraph = tf.Graph()
    with self.graph.as_default():
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      self.sess = tf.Session(config=config)
    with self.testgraph.as_default():
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.operation_timeout_in_ms=1000
      self.testsess = tf.Session(config=config)
    self.build_model()
    data = open(FLAGS.data_path).readlines()
    size = len(data)
    self.read_data(data[:int(0.8*size)])
    self.test_read_data(data[int(0.8*size):])
    self.id2word = json.loads(open('id2word_qald5311_debayan_'+FLAGS.modelnum+'.txt').read())
    

  def read_data(self, recs):
    inputs = []
    enc_input_weights = []
    outputs = []
    dec_input_weights = []
    
    for rec in recs:
      inp, outp = rec.strip().split(' output ')
      inp = inp.split(' ')
      print(len(inp))
      outp = outp.split(' ')

      enc_input = []
      for t in inp:
        enc_input.append(int(t))
      enc_input_len = len(enc_input)
      enc_input += [0]*((FLAGS.max_input_sequence_len-enc_input_len)) 
      enc_input = np.array(enc_input).reshape([-1,1])
      inputs.append(enc_input)
      weight = np.zeros(FLAGS.max_input_sequence_len)
      weight[:enc_input_len]=1
      enc_input_weights.append(weight)
 
      output=[pointer_net.START_ID]
      for i in outp:
        # Add 2 to value due to the sepcial tokens
        try:
            output.append(int(i)+2)
        except:
            pass
      output.append(pointer_net.END_ID)
      dec_input_len = len(output)-1
      output += [pointer_net.PAD_ID]*(FLAGS.max_output_sequence_len-dec_input_len)
      output = np.array(output)
      outputs.append(output)
      weight = np.zeros(FLAGS.max_output_sequence_len)
      weight[:dec_input_len]=1
      dec_input_weights.append(weight)
      
    self.inputs = np.stack(inputs)
    self.enc_input_weights = np.stack(enc_input_weights)
    self.outputs = np.stack(outputs)
    self.dec_input_weights = np.stack(dec_input_weights)
    print("Load inputs:            " +str(self.inputs.shape))
    print("Load enc_input_weights: " +str(self.enc_input_weights.shape))
    print("Load outputs:           " +str(self.outputs.shape))
    print("Load dec_input_weights: " +str(self.dec_input_weights.shape))
  
  def test_read_data(self,recs):
    inputs = []
    enc_input_weights = []
    outputs = []
    dec_input_weights = []

    for rec in recs:
      inp, outp = rec.strip().split(' output ')
      inp = inp.split(' ')
      outp = outp.split(' ')

      enc_input = []
      for t in inp:
        enc_input.append(int(t))
      enc_input_len = len(enc_input)
      enc_input += [0]*((FLAGS.max_input_sequence_len-enc_input_len))
      enc_input = np.array(enc_input).reshape([-1,1])
      inputs.append(enc_input)
      weight = np.zeros(FLAGS.max_input_sequence_len)
      weight[:enc_input_len]=1
      enc_input_weights.append(weight)

      output=[pointer_net.START_ID]
      for i in outp:
        # Add 2 to value due to the sepcial tokens
        try:
            output.append(int(i)+2)
        except:
            pass
      output.append(pointer_net.END_ID)
      dec_input_len = len(output)-1
      output += [pointer_net.PAD_ID]*(FLAGS.max_output_sequence_len-dec_input_len)
      output = np.array(output)
      outputs.append(output)
      weight = np.zeros(FLAGS.max_output_sequence_len)
      weight[:dec_input_len]=1
      dec_input_weights.append(weight)

    self.test_inputs = np.stack(inputs)
    self.test_enc_input_weights = np.stack(enc_input_weights)
    self.test_outputs = np.stack(outputs)
    self.test_dec_input_weights = np.stack(dec_input_weights)
    print("Load inputs:            " +str(self.test_inputs.shape))
    print("Load enc_input_weights: " +str(self.test_enc_input_weights.shape))
    print("Load outputs:           " +str(self.test_outputs.shape))
    print("Load dec_input_weights: " +str(self.test_dec_input_weights.shape))

  def get_batch(self):
    data_size = self.inputs.shape[0]
    sample = np.random.choice(data_size,FLAGS.batch_size,replace=True)
    return self.inputs[sample],self.enc_input_weights[sample],\
      self.outputs[sample], self.dec_input_weights[sample]

  def get_test_batch(self):
    #data_size = self.test_inputs.shape[0]
    #sample = range(FLAGS.batch_size)#np.random.choice(data_size,FLAGS.batch_size,replace=True)
    return self.test_inputs,self.test_enc_input_weights,\
      self.test_outputs, self.test_dec_input_weights

  def build_model(self):
    with self.testgraph.as_default():
      self.testmodel = pointer_net.PointerNet(batch_size=1,
                    max_input_sequence_len=FLAGS.max_input_sequence_len,
                    max_output_sequence_len=FLAGS.max_output_sequence_len,
                    rnn_size=FLAGS.rnn_size,
                    attention_size=FLAGS.attention_size,
                    num_layers=FLAGS.num_layers,
                    beam_width=FLAGS.beam_width,
                    learning_rate=FLAGS.learning_rate,
                    max_gradient_norm=FLAGS.max_gradient_norm,
                    forward_only=True)

    with self.graph.as_default():
      # Build model
      self.model = pointer_net.PointerNet(batch_size=FLAGS.batch_size, 
                    max_input_sequence_len=FLAGS.max_input_sequence_len, 
                    max_output_sequence_len=FLAGS.max_output_sequence_len, 
                    rnn_size=FLAGS.rnn_size, 
                    attention_size=FLAGS.attention_size, 
                    num_layers=FLAGS.num_layers,
                    beam_width=FLAGS.beam_width, 
                    learning_rate=FLAGS.learning_rate, 
                    max_gradient_norm=FLAGS.max_gradient_norm, 
                    forward_only=self.forward_only)
      # Prepare Summary writer
      self.writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',self.sess.graph)
      # Try to get checkpoint
      ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
      if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Load model parameters from %s" % ckpt.model_checkpoint_path)
        self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
      else:
        print("Created model with fresh parameters.")
        self.sess.run(tf.global_variables_initializer())


  def train(self):
    step_time = 0.0
    loss = 0.0
    current_step = 0
    bestfuzz = 0
    while True:
      start_time = time.time()
      inputs,enc_input_weights, outputs, dec_input_weights = \
                  self.get_batch()
      summary, step_loss, predicted_ids_with_logits, targets, debug_var = \
                  self.model.step(self.sess, inputs, enc_input_weights, outputs, dec_input_weights)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      #DEBUG PART
      #print("debug")
      #print(debug_var)
      #return
      #/DEBUG PART

      #Time to print statistic and save model
      if current_step % FLAGS.steps_per_checkpoint == 0:
        with self.sess.as_default():
          gstep = self.model.global_step.eval()
        print ("global step %d step-time %.2f loss %.2f" % (gstep, step_time, loss))
        #Write summary
        self.writer.add_summary(summary, gstep)
        checkpoint_path = os.path.join(FLAGS.log_dir, "convex_hull.ckpt")
        self.model.saver.save(self.sess, checkpoint_path, global_step=self.model.global_step)
        step_time, loss = 0.0, 0.0

        #Randomly choose one to check
        #sample = np.random.choice(FLAGS.batch_size,1)[0]
        #print("="*20)
        #print("Predict: "+str(np.array(predicted_ids_with_logits[1][sample]).reshape(-1)))
        #print("Target : "+str(targets[sample]))
        #print("="*20) 
        # Prepare Summary writer
        # Try to get checkpoint
        ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
          print("Load model parameters from %s" % ckpt.model_checkpoint_path)
          self.testmodel.saver.restore(self.testsess, ckpt.model_checkpoint_path) 
          fuzz = self.eval()
          print("fuzz: ",fuzz)
          if fuzz > bestfuzz:
            print(fuzz," better than ",bestfuzz)
            bestfuzz = fuzz
            self.model.saver.save(self.sess, os.path.join(FLAGS.log_dir+'/solid/', "convex_hull_%f.ckpt"%(fuzz)), global_step=self.model.global_step) 
        
  def eval(self):
    """ Randomly get a batch of data and output predictions """  
    inputs,enc_input_weights, outputs, dec_input_weights = self.get_test_batch()
    avgfuzz = 0
    rat = 0
    count = 0
    for input_,enc_input_weights_,outputs_,dec_input_weights_ in zip(inputs,enc_input_weights,outputs,dec_input_weights):
      try:
        predicted_ids = self.testmodel.step(self.testsess, [input_], [enc_input_weights_])
      except Exception as err:
        print(err)
        continue
      count += 1
      #print("="*20)
      #print("inputs: ",input_)
      query = []
      for x in input_:
        if x[0] == -1:
          break
        query.append(self.id2word[str(x[0])])   
      print("input query: ", ' '.join(query))
      #print("* %dth sample target: %s" % (i,str(outputs_-2)))
      goldquery = []
      for x in outputs_-2:
        if x > 0:
          goldquery.append(self.id2word[str(input_[x-1][0])]) 
      gold = ' '.join(goldquery)
      print("goldtarget: ", ' '.join(goldquery))   
      predict = predicted_ids[0][0]
      #print("predict:",predict)
      predquery = []
      for x in predict:
        if x > 0:
          predquery.append(self.id2word[str(input_[x-1][0])])
      ans = ' '.join(predquery)
      print("prediction: ", ' '.join(predquery))       
      rat += fuzz.ratio(ans,gold)
      avgfuzz = rat/float(count)
      print("avg fuzz: ", avgfuzz)
      #print("="*20)
    return avgfuzz

  def run(self):
    if self.forward_only:
      self.eval()
    else:
      self.train()

def main(_):
  convexHull = ConvexHull(FLAGS.forward_only)
  convexHull.run()

if __name__ == "__main__":
  tf.app.run()
