import tensorflow as tf
import glob
import json
import sys
from data_helper import Vocab, Data_Helper
from vectorisergold import Vectoriser

v = Vectoriser('wikidataprops.json')



def example_generator(filename, vocab_path, vocab_size, max_enc_len, max_dec_len, training=False):
        vocab = Vocab(vocab_path, vocab_size)
        d = json.loads(open(filename).read())

        for item in d:
                if not item["question"] or not item["sparql_wikidata"]:
                        continue
                question = item["question"]
                intermediate_sparql = item["sparql_wikidata"].replace(","," , ").replace('{',' { ').replace('}',' } ')
                uid = item["uid"]
                start_decoding = vocab.word_to_id(vocab.START_DECODING)
                stop_decoding = vocab.word_to_id(vocab.STOP_DECODING)
                 
                questiontokens,questionvectors = v.vectorise(question, intermediate_sparql)

                enc_input = questionvectors[:max_enc_len]
                enc_len = len(enc_input)
                if enc_len == 0:
                        continue
                question_words = [w.lower() for w in questiontokens][:max_enc_len]
                #enc_len = len(question_words)
                enc_input_mask = [vocab.word_to_id(w) for w in question_words]
                enc_input_extend_vocab, question_oovs = Data_Helper.article_to_ids(question_words, vocab)

                intsparql_words_ = intermediate_sparql.replace('wd:','').replace('wdt:','').replace('ps:','').replace('pq:','').replace('p:','').lower().split()
                intsparql_words = [x for x in intsparql_words_]
                intsparql_ids = [vocab.word_to_id(w) for w in intsparql_words]
                intsparql_ids_extend_vocab = Data_Helper.abstract_to_ids(intsparql_words, vocab, question_oovs)
               
                dec_input, target = Data_Helper.get_dec_inp_targ_seqs(intsparql_ids, max_dec_len, start_decoding, stop_decoding)
                _, target = Data_Helper.get_dec_inp_targ_seqs(intsparql_ids_extend_vocab, max_dec_len, start_decoding, stop_decoding)

#                print("sprql: ",intermediate_sparql)
#                print("intsparql_words: ",intsparql_words)
#                print("sprqlids: ",intsparql_ids)
#                print("extendvocab: ",intsparql_ids_extend_vocab)
#                print("decinput: ", dec_input)
#                print("target: ",target)
                dec_len = len(dec_input)
         
                output = {
                        "uid":uid,
                        "enc_len":enc_len,
                        "enc_input" : enc_input,
                        "enc_input_mask": enc_input_mask,
                        "enc_input_extend_vocab"  : enc_input_extend_vocab,
                        "question_oovs" : question_oovs,
                        "dec_input" : dec_input,
                        "target" : target,
                        "dec_len" : dec_len,
                        "intermediate_sparql" : intermediate_sparql
                }

                yield output

def batch_generator(generator, filenames, vocab_path,  vocab_size, max_enc_len, max_dec_len, batch_size, training):
        dataset = tf.data.Dataset.from_generator(generator, args = [filenames, vocab_path,  vocab_size, max_enc_len, max_dec_len, training],
                                                                                        output_types = {
                                                                                                "uid":tf.int32,
                                                                                                "enc_len":tf.int32,
                                                                                                "enc_input" : tf.float32,
                                                                                                "enc_input_mask" : tf.int32,
                                                                                                "enc_input_extend_vocab"  : tf.int32,
                                                                                                "question_oovs" : tf.string,
                                                                                                "dec_input" : tf.int32,
                                                                                                "target" : tf.int32,
                                                                                                "dec_len" : tf.int32,
                                                                                                "intermediate_sparql" : tf.string
                                                                                        }, output_shapes={
                                                                                                "uid": [],
                                                                                                "enc_len":[],
                                                                                                "enc_input" : [None,None],
                                                                                                "enc_input_mask" : [None],
                                                                                                "enc_input_extend_vocab"  : [None],
                                                                                                "question_oovs" : [None],
                                                                                                "dec_input" : [None],
                                                                                                "target" : [None],
                                                                                                "dec_len" : [],
                                                                                                "intermediate_sparql" : []
                                                                                        })
        dataset = dataset.padded_batch(batch_size, padded_shapes=({"enc_len":[],
                                                                                                "enc_input" : [None,None],
                                                                                                "enc_input_mask" : [None],
                                                                                                "uid": [],
                                                                                                "enc_input_extend_vocab"  : [None],
                                                                                                "question_oovs" : [None],
                                                                                                "dec_input" : [max_dec_len],
                                                                                                "target" : [max_dec_len],
                                                                                                "dec_len" : [],
                                                                                                "intermediate_sparql" : []
                                                                                                }),
                                                                                        padding_values={"enc_len":-1,
                                                                                                "uid": -1,
                                                                                                "enc_input" : -1.0,
                                                                                                "enc_input_mask" : -1,
                                                                                                "enc_input_extend_vocab"  : 1,
                                                                                                "question_oovs" : b'',
                                                                                                "dec_input" : 1,
                                                                                                "target" : 1,
                                                                                                "dec_len" : -1,
                                                                                                "intermediate_sparql" : b""},
                                                                                        drop_remainder=True)
        def update(entry):
                return ({"enc_input" : entry["enc_input"],
                         "enc_input_mask" : entry["enc_input_mask"],
                         "uid" : entry["uid"],
                        "extended_enc_input" : entry["enc_input_extend_vocab"],
                        "question_oovs" : entry["question_oovs"],
                        "enc_len" : entry["enc_len"],
                        "max_oov_len" : tf.shape(entry["question_oovs"])[1] },

                        {"dec_input" : entry["dec_input"],
                        "dec_target" : entry["target"],
                        "dec_len" : entry["dec_len"],
                        "intermediate_sparql" : entry["intermediate_sparql"]})


        dataset = dataset.map(update)

        return dataset


def entitybatcher(data_path, vocab_path, hpm):
  
        print(data_path)
        dataset = batch_generator(example_generator, data_path, vocab_path, hpm["vocab_size"], hpm["max_enc_len"], hpm["max_dec_len"], hpm["batch_size"], hpm["training"] )
        return dataset

