import tensorflow as tf
from tensorflow.python.training import training_util
from training_helper import train_model
from predict_helper import predict
from entitybatcher import entitybatcher
from transformer import Transformer
import os
import random
import sys
import json



def my_model(features, labels, mode, params):
        
        predictions, attn_weights = predict(features, params, transformer)
        estimator_spec = tf.estimator.EstimatorSpec(mode,  predictions={"predictions":predictions})
  
        print(transformer.summary())
        return estimator_spec


def build_model(params):

        config = tf.estimator.RunConfig(
                tf_random_seed=params["seed"], 
                log_step_count_steps=params["log_step_count_steps"],
                save_summary_steps=params["save_summary_steps"]
        )

        return tf.estimator.Estimator(
                                        model_fn=my_model,
                                        params=params, config=config, model_dir=params["model_dir"] )


def train(params):
        assert params["training"], "change training mode to true"

        tf.compat.v1.logging.info("Building the model ...")
        transformer = Transformer(
                num_layers=params["num_layers"], d_model=params["model_depth"], num_heads=params["num_heads"], dff=params["dff"], 
                vocab_size=params["vocab_size"], batch_size=params["batch_size"])

        tf.compat.v1.logging.info("Creating the batcher ...")
        ids = [x for x in range(1,3254)]
        length = int(len(ids)/5) #length of each fold
        folds = []
        for i in range(5):
            folds += [ids[i*length:(i+1)*length]]
        folds += [ids[5*length:len(ids)]]
        testids = folds[params['fold']-1]
        trainids_ = []
        for i in range(5):
            if params['fold'] - 1 == i:
                continue
            trainids_ += folds[i]
        trainids = trainids_#[:2275]
        #devids = trainids_[2275:]
        print("fold:",params['fold'])
        print("trainids:", trainids,len(trainids))
        print("testids:",testids,len(testids))
       # print("devids:",devids,len(devids)) 
        b = entitybatcher(params["data_dir"], params["vocab_path"], params, trainids) #curricullum 1
        devb = None#entitybatcher(params["data_dir"], params["vocab_path"],params, devids)
        testb = entitybatcher(params["data_dir"], params["vocab_path"],params, testids)

        tf.compat.v1.logging.info("Creating the checkpoint manager")
        logdir = "{}/logdir".format(params["model_dir"])
        summary_writer = tf.summary.create_file_writer(logdir)
        checkpoint_dir = "{}/checkpoint".format(params["model_dir"])
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), transformer=transformer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=11)

        ckpt.restore(ckpt_manager.latest_checkpoint)
        if ckpt_manager.latest_checkpoint:
                print("Restored from {}".format(ckpt_manager.latest_checkpoint))
        else:
                print("Initializing from scratch.")

        tf.compat.v1.logging.info("Starting the training ...")
        train_model(transformer, b, devb, testb, params, ckpt, ckpt_manager, summary_writer, trainids)
 


def eval(model, params):
        pass


def test(params):
        ids = [x for x in range(1,3254)]
        length = int(len(ids)/5) #length of each fold
        folds = []
        for i in range(5):
            folds += [ids[i*length:(i+1)*length]]
        folds += [ids[5*length:len(ids)]]
        testids = folds[params['fold']-1]
        trainids_ = []
        for i in range(5):
            if params['fold'] - 1 == i:
                continue
            trainids_ += folds[i]
        trainids = trainids_#[:2275]
        #devids = trainids_[2275:]
        print("fold:",params['fold'])
        #print("trainids:", trainids,len(trainids))
        #print("testids:",testids,len(testids))

        assert not params["training"], "change training mode to false"
        checkpoint_dir = "{}/checkpoint".format(params["model_dir"])
        logdir = "{}/logdir".format(params["model_dir"])
        model = Transformer( num_layers=params["num_layers"], d_model=params["model_depth"], num_heads=params["num_heads"], dff=params["dff"], vocab_size=params["vocab_size"], batch_size=params["batch_size"])
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), transformer=model)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=11)
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
        out,att,retarr = predict(entitybatcher(params["data_dir"], params["vocab_path"], params,testids), params, model)
        f = open(params["model_dir"].strip("/")+'out.json','w')
        f.write(json.dumps(retarr,indent=4,sort_keys=True))
        f.close()
