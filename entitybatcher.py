import tensorflow as tf
import glob
import json
from data_helper import Vocab, Data_Helper


def example_generator(filename, vocab_path, vocab_size, max_enc_len, max_dec_len, training=False):

        vocab = Vocab(vocab_path, vocab_size)

        d = json.loads(open(filename).read())

        for item in d:
                if not item["question"] or not item["intermediate_sparql"]:
                        continue
                question = item["question"]
                intermediate_sparql = item["intermediate_sparql"]

                start_decoding = vocab.word_to_id(vocab.START_DECODING)
                stop_decoding = vocab.word_to_id(vocab.STOP_DECODING)

                question_words_ = question.replace('?','').split()[ : max_enc_len] #handle cases like "who is older than 25?"  Here 25? fails to convert to int
                question_words = [x.lower() for x in question_words_]
                
                enc_len = len(question_words)
                enc_input = [vocab.word_to_id(w) for w in question_words]
                enc_input_extend_vocab, question_oovs = Data_Helper.article_to_ids(question_words, vocab)

                #intsparql_sentences = [intermediate_sparql]
                #abstract = ' '.join(abstract_sentences)
                intsparql_words_ = intermediate_sparql.replace(","," , ").split()
                intsparql_words = [x.lower() for x in intsparql_words_]
                intsparql_ids = [vocab.word_to_id(w) for w in intsparql_words]
                intsparql_ids_extend_vocab = Data_Helper.abstract_to_ids(intsparql_words, vocab, question_oovs)
                dec_input, target = Data_Helper.get_dec_inp_targ_seqs(intsparql_ids, max_dec_len, start_decoding, stop_decoding)
                _, target = Data_Helper.get_dec_inp_targ_seqs(intsparql_ids_extend_vocab, max_dec_len, start_decoding, stop_decoding)
                dec_len = len(dec_input)

                output = {
                        "enc_len":enc_len,
                        "enc_input" : enc_input,
                        "enc_input_extend_vocab"  : enc_input_extend_vocab,
                        "question_oovs" : question_oovs,
                        "dec_input" : dec_input,
                        "target" : target,
                        "dec_len" : dec_len,
                        "question" : question,
                        "intermediate_sparql" : intermediate_sparql,
                        "intsparql_sents" : [intermediate_sparql]
                }


                yield output


def batch_generator(generator, filenames, vocab_path, vocab_size, max_enc_len, max_dec_len, batch_size, training):

        dataset = tf.data.Dataset.from_generator(generator, args = [filenames, vocab_path, vocab_size, max_enc_len, max_dec_len, training],
                                                                                        output_types = {
                                                                                                "enc_len":tf.int32,
                                                                                                "enc_input" : tf.int32,
                                                                                                "enc_input_extend_vocab"  : tf.int32,
                                                                                                "question_oovs" : tf.string,
                                                                                                "dec_input" : tf.int32,
                                                                                                "target" : tf.int32,
                                                                                                "dec_len" : tf.int32,
                                                                                                "question" : tf.string,
                                                                                                "intermediate_sparql" : tf.string,
                                                                                                "intsparql_sents" : tf.string
                                                                                        }, output_shapes={
                                                                                                "enc_len":[],
                                                                                                "enc_input" : [None],
                                                                                                "enc_input_extend_vocab"  : [None],
                                                                                                "question_oovs" : [None],
                                                                                                "dec_input" : [None],
                                                                                                "target" : [None],
                                                                                                "dec_len" : [],
                                                                                                "question" : [],
                                                                                                "intermediate_sparql" : [],
                                                                                                "intsparql_sents" : [None]
                                                                                        })
        dataset = dataset.padded_batch(batch_size, padded_shapes=({"enc_len":[],
                                                                                                "enc_input" : [None],
                                                                                                "enc_input_extend_vocab"  : [None],
                                                                                                "question_oovs" : [None],
                                                                                                "dec_input" : [max_dec_len],
                                                                                                "target" : [max_dec_len],
                                                                                                "dec_len" : [],
                                                                                                "question" : [],
                                                                                                "intermediate_sparql" : [],
                                                                                                "intsparql_sents" : [None]}),
                                                                                        padding_values={"enc_len":-1,
                                                                                                "enc_input" : 1,
                                                                                                "enc_input_extend_vocab"  : 1,
                                                                                                "question_oovs" : b'',
                                                                                                "dec_input" : 1,
                                                                                                "target" : 1,
                                                                                                "dec_len" : -1,
                                                                                                "question" : b"",
                                                                                                "intermediate_sparql" : b"",
                                                                                                "intsparql_sents" : b''},
                                                                                        drop_remainder=True)
        def update(entry):
                return ({"enc_input" : entry["enc_input"],
                        "extended_enc_input" : entry["enc_input_extend_vocab"],
                        "question_oovs" : entry["question_oovs"],
                        "enc_len" : entry["enc_len"],
                        "question" : entry["question"],
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
