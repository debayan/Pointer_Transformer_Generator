import tensorflow as tf
from utils import create_masks
import time
import sys
from data_helper import Data_Helper, Vocab
import time
from fuzzywuzzy import fuzz
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np

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
                print("learning rate: ", tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2))
                return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(loss_object, real, pred):
#        mask = tf.math.logical_not(tf.math.equal(real, 0))
        #loss_ = scce_with_ls(pred,real)
        loss_ = loss_object(real, pred)

#        mask = tf.cast(mask, dtype=loss_.dtype)
#        loss_ *= mask

        return tf.reduce_mean(loss_)


def train_step(features, labels, params, model, optimizer, loss_object, train_loss_metric, batchcount, testbatcher):
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
        if batchcount%100 == 0 and batchcount > 1:
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
#                        if not predblock:
#                            predblock = predictions
#                        else:
#                            predblock = tf.tile([predblock,predictions],axis=2)
                        predblock = predictions
                        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
                        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
                        # concatentate the predicted_id to the output which is given to the decoder
                        # as its input.
                        output = tf.concat([output, predicted_id], axis=-1)
                        
                    #beam search
#                    predblocktf = tf.transpose(predblock, [1, 0, 2])
#                    decoded, logprobs = tf.nn.ctc_beam_search_decoder(predblocktf, sequence_length=tf.constant(params["batch_size"]*[128]) ,beam_width=100,top_paths=10)
#                    outputdict = {}
#                    for pathnumber in range(10):
#                        indices = np.delete(decoded[pathnumber].indices.numpy(),1,axis=1)
#                        values = decoded[pathnumber].values.numpy()
#                        for idx,val in zip(indices,values):
#                            if idx[0] not in outputdict:
#                                outputdict[idx[0]] = [[],[],[],[],[],[],[],[],[],[]]
#                            outputdict[idx[0]][pathnumber].append(val)
#                    print("outputdict: ",outputdict[0])
#                    print("outputdict: ",outputdict[1])
#                    sys.exit(1)
                                            
                    idx = 0
                    for question,nonbeamanswer,target,uid,oov in zip(testfeatures["question"],output,testlabels["dec_target"],testfeatures["uid"],testfeatures['question_oovs']):
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
#                        for pathnum in range(10):
#                            words = []
#                            for x in answer[pathnum]:
#                                if x==3 or x==1:
#                                    break
#                                if x < vocab.size():
#                                    words.append(vocab.id_to_word(x))
#                                else:
#                                    words.append(list(oov.numpy())[x - vocab.size()].decode('utf-8'))
#                            answer_ = ' '.join(words)
#                            print("answer: ", answer_)
                        words = []
                        for x in list(nonbeamanswer[1:].numpy()):
                            if x==3 or x==1:
                                break
                            if x < vocab.size():
                                words.append(vocab.id_to_word(x))
                            else:
                                words.append(list(oov.numpy())[x - vocab.size()].decode('utf-8'))
                        nonbeamanswer_ = ' '.join(words)
                        qcount += 1
                        #totalfuzz += fuzz.ratio(target_.lower(), answer_.lower())
                        totalfuzznonbeam += fuzz.ratio(target_.lower(), nonbeamanswer_.lower())
                        
                        #print("raw targ: ", list(target.numpy()))
                        #print("raw answer: ",list(answer.numpy()))
                        print("target: ", target_)
                        #print("answer: ", answer_)
                        print("nonbeamanswer: ", nonbeamanswer_)
                        #print("avg fuzz after %d questions = %f"%(qcount,float(totalfuzz)/qcount))
                        print("nonbeam avg fuzz after %d questions = %f"%(qcount,float(totalfuzznonbeam)/qcount))
                    print("testidx: ",testidx)
                    if testidx >= 2:
                        break
                except Exception as err:
                        print("er: ",err)             
        #return testlossfloat
        return totalfuzznonbeam/100.0,qcount

def train_model(model, batcher, testbatcher, params, ckpt, ckpt_manager):
        learning_rate = CustomSchedule(params["model_depth"])
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=0.1)  
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
                                if idx%100 == 0 and idx > 1:
                                    print("valfuzz - bestfuzz : ",valfuzz,bestfuzz)
                                    if valfuzz > bestfuzz:
                                        bestfuzz = valfuzz
                                        print("Best val fuzz so far: %f"%(valfuzz/qcount))
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
