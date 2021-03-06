import tensorflow as tf
import argparse
from build_eval_test import build_model, train, test
import logging
import os

tf.config.threading.set_intra_op_parallelism_threads(10)
tf.config.threading.set_inter_op_parallelism_threads(10)

def main():

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

        parser = argparse.ArgumentParser()
        parser.add_argument("--max_enc_len", default=128, help="Encoder input max sequence length", type=int)
        
        parser.add_argument("--max_dec_len", default=128, help="Decoder input max sequence length", type=int)
        
        parser.add_argument("--batch_size", default=16, help="batch size", type=int)
        
        parser.add_argument("--vocab_size", default=200, help="Vocabulary size", type=int)
        
        parser.add_argument("--num_layers", default=2, help="Model encoder and decoder number of layers", type=int)
        

        parser.add_argument("--model_depth", default=500, help="Model Embedding size", type=int)
        
        parser.add_argument("--num_heads", default=10, help="Multi Attention number of heads", type=int)
        
        parser.add_argument("--dff", default=2048, help="Dff", type=int)

        parser.add_argument("--seed", default=123, help="Seed", type=int)
        
        parser.add_argument("--log_step_count_steps", default=1, help="Log each N steps", type=int)
        
        parser.add_argument("--max_steps",default=15000, help="Max steps for training", type=int)
                
        parser.add_argument("--save_summary_steps", default=1000, help="Save summaries every N steps", type=int)
        
        parser.add_argument("--checkpoints_save_steps", default=0, help="Save checkpoints every N steps", type=int)
        
        parser.add_argument("--mode", help="training, eval or test options")

        parser.add_argument("--model_dir", help="Model folder")

        parser.add_argument("--data_dir",  help="Data Folder")

        parser.add_argument("--vocab_path", help="Vocab path")
        
        parser.add_argument("--prop_embedding_path", default="wikidatapropembeddings.json" ,help="Wikidata property embedding file path")

        parser.add_argument("--test_dir", help="Test file input")

        
        args = parser.parse_args()
        params = vars(args)
        print(params)

        assert params["mode"], "mode is required. train, test or eval option"
        if params["mode"] == "train":
                params["training"] = True ; params["eval"] = False ; params["test"] = False
        elif params["mode"] == "eval":
                params["training"] = False ; params["eval"] = True ; params["test"] = False
        elif params["mode"] == "test":
                params["training"] = False ; params["eval"] = False ; params["test"] = True;
        else:
                raise NameError("The mode must be train , test or eval")
        assert os.path.exists(params["data_dir"]), "data_dir doesn't exist"
        assert os.path.isfile(params["vocab_path"]), "vocab_path doesn't exist"

   
        if params["training"]:
              
                train( params)
                
        elif params["eval"]:
                print("test")
                test(params)
        elif not params["training"]:
                pass


if __name__ == "__main__":
        main()
