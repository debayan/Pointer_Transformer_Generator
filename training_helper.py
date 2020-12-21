import tensorflow as tf
from utils import create_masks
import time
import sys
from data_helper import Vocab
import time
from fuzzywuzzy import fuzz

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
                super(CustomSchedule, self).__init__()

                self.d_model = d_model
                self.d_model = tf.cast(self.d_model, tf.float32)

                self.warmup_steps = warmup_steps

        def __call__(self, step):
                arg1 = tf.math.rsqrt(step)
                arg2 = step * (self.warmup_steps ** -1.5)

                return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(loss_object, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)


def train_step(features, labels, params, model, optimizer, loss_object, train_loss_metric, batchcount, testbatcher):
  
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(features["enc_input"], labels["dec_input"])
        testlossfloat = 999.0
        with tf.GradientTape() as tape:
                output, attn_weights = model(features["enc_input"],features["extended_enc_input"], features["max_oov_len"], labels["dec_input"], training=params["training"], 
                                                                        enc_padding_mask=enc_padding_mask, 
                                                                        look_ahead_mask=combined_mask,
                                                                        dec_padding_mask=dec_padding_mask)
                #print("input: ",features)
#                print("target: ",labels["dec_target"])
#                print("target shape: ", labels["dec_target"].shape)
                #print("output: ",output)
#                print("output shape: ",output.shape)
                loss = loss_function(loss_object, labels["dec_target"], output)
                qcount = 0
                totalfuzz = 0.0
                if batchcount%100 == 0 and batchcount > 1:
                    vocab = Vocab(params['vocab_path'], params['vocab_size'])
                    for testidx,testbatch in enumerate(testbatcher):
                        testfeatures = testbatch[0]
                        testlabels = testbatch[1]
                        test_enc_padding_mask, test_combined_mask, test_dec_padding_mask = create_masks(testfeatures["enc_input"], testlabels["dec_input"])
                        testoutput, test_attn_weights = model(testfeatures["enc_input"],testfeatures["extended_enc_input"], testfeatures["max_oov_len"], testlabels["dec_input"], training=False, enc_padding_mask=test_enc_padding_mask, look_ahead_mask=test_combined_mask,dec_padding_mask=test_dec_padding_mask)
                        for answer,target in zip(testoutput,testlabels["dec_target"]):
                            qcount += 1
#                        #print("target: ",[x for x in list(target.numpy()) if x != 1 and x!=3])
#                        #print("answer: ",[x for x in list(tf.math.argmax(answer, axis=1).numpy()) if x != 1 and x!= 3])
                            target = ' '.join([vocab.id_to_word(x) for x in list(target.numpy()) if x != 1 and x != 3])
                            answer = ' '.join([vocab.id_to_word(x) for x in list(tf.math.argmax(answer, axis=1).numpy()) if x != 1 and x!= 3])
                            print("target: ", target)
                            print("answer: ", answer)
                            totalfuzz += fuzz.ratio(target,answer)
                        break
                            #test_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
                            #testloss = loss_function(test_loss_object, testlabels["dec_target"],testoutput)
                        #print("val loss = ",float(testloss))
                        #testlossfloat = float(testloss)
                        #time.sleep(5)
                        #break


        gradients = tape.gradient(loss, model.trainable_variables)    
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss_metric(loss)
        #return testlossfloat
        return totalfuzz/100.0,qcount

def train_model(model, batcher, testbatcher, params, ckpt, ckpt_manager):
        learning_rate = CustomSchedule(params["model_depth"])
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)  
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
        train_loss_metric = tf.keras.metrics.Mean(name="train_loss_metric")

        try:
                bestfuzz = 0.0
                epoch = 0
                while int(ckpt.step) < params["max_steps"]:
                        epoch += 1
                        for idx,batch in enumerate(batcher):
                                t0 = time.time()
                                valfuzz,qcount = train_step(batch[0], batch[1], params, model, optimizer, loss_object, train_loss_metric, idx, testbatcher)
                                t1 = time.time()
                                if idx%100 == 0:
                                    print("valfuzz - bestfuzz : ",valfuzz,bestfuzz)
                                    if valfuzz > bestfuzz:
                                        bestfuzz = valfuzz
                                        print("Best val fuzz so far: %f"%(valfuzz/qcount))
                                        ckpt_manager.save(checkpoint_number=int(ckpt.step))
                                        print("Saved checkpoint for step {}".format(int(ckpt.step)))
                                    print("epoch {} step {}, time : {}, loss: {}".format(epoch,int(ckpt.step), t1-t0, train_loss_metric.result()))
                                #if int(ckpt.step) % params["checkpoints_save_steps"] ==0 :
                                #       ckpt_manager.save(checkpoint_number=int(ckpt.step))
                                #        print("Saved checkpoint for step {}".format(int(ckpt.step)))
                                ckpt.step.assign_add(1)
                        
        except KeyboardInterrupt:
                ckpt_manager.save(int(ckpt.step))
                print("Saved checkpoint for step {}".format(int(ckpt.step)))
        except Exception as err:
                print(err)
