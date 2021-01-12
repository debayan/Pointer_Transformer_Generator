import tensorflow as tf
import glob
import json
import sys
from data_helper import Vocab, Data_Helper
import re
from elasticsearch import Elasticsearch
import gensim
import numpy as np

es = Elasticsearch(host="ltcpu1",port=32816)
entembedcache = {}

print("loading fasttext")
fasttext = gensim.models.KeyedVectors.load_word2vec_format('./data/fasttext-wiki-news-subwords-300')
print("loaded fastext")




def example_generator(filename, vocab_path, vocab_size, max_enc_len, max_dec_len, training=False):
        vocab = Vocab(vocab_path, vocab_size)
        d = json.loads(open(filename).read())

        for item in d:
                if not item["question"] or not item["intermediate_sparql"]:
                        continue
                if len(item["question"].split()) > 80:
                        continue
                question = item["question"].replace('?','').replace('{','').replace('}','').lower()
                intermediate_sparql = item["intermediate_sparql"]
                sparql = item['sparql_wikidata'].lower().replace('{',' { ').replace('}',' } ')
                ents = re.findall( r'wd:(.*?) ', sparql)
                rels = re.findall( r'wdt:(.*?) ', sparql)
                rels += re.findall ( r'ps:(.*?) ', sparql)
                rels += re.findall ( r'pq:(.*?) ', sparql)
                rels += re.findall ( r'p:(.*?) ', sparql)
                #print(sparql)
                #print(ents,rels)
                entrels = []
                entrelsembeddings = []
                for ent in ents:
                        ent = ent.upper().replace('.','')
                        enturl = '<http://www.wikidata.org/entity/'+ent+'>'
#                        ent=ent.lower().replace('.','')
                        
                        vector = []
                        try:
                                res = es.search(index="wikidataentitylabelindex01", body={"query":{"term":{"uri":{"value":'http://wikidata.dbpedia.org/resource/'+ent}}},"size":1})
                                #print(res)
                                label = res['hits']['hits'][0]['_source']['wikidataLabel']
                                #print("label: ",label)
                                ftv = []
                                for word in label.split():
                                #     print(word,fasttext.word_vec(word))
                                     ftv.append(fasttext.word_vec(word))
                                labelfasttextvector = np.average(ftv,axis=0)
                                #print("avg : ",labelfasttextvector)
                                vector = list(labelfasttextvector)
                                #print("foundlabel")
                        except Exception as err:
                                vector = 300*[0.0]
                                #print(ent," err ",err)
                                
                        try:
                                res = es.search(index="wikidataembedsindex01", body={"query":{"term":{"key":{"value":enturl}}}})
                                entityembedding = [float(x) for x in res['hits']['hits'][0]['_source']['embedding']]
                                vector += entityembedding
                                #print("foundentity embedding")
                        except Exception as err:
                                vector += 200*[-1.0]
                                #print(ent," err ",err)
                        ent = ent.lower()
                        entrels.append(ent)
                        entrelsembeddings.append(vector)
                        #print(ent,vector)
                        
                for rel in rels:
                        rel = rel.upper()
                        relurl = '<http://www.wikidata.org/entity/'+rel+'>'
                        rel=rel.lower()
                        res = es.search(index="wikidataembedsindex01", body={"query":{"term":{"key":{"value":relurl}}}})
                        try:
                                embedding = [float(x) for x in res['hits']['hits'][0]['_source']['embedding']]
                                entrels.append(rel)
                                entrelsembeddings.append(300*[0.0]+embedding)
                                #print(relurl,embedding)
                        except Exception as e:
                                entrels.append(rel)
                                entrelsembeddings.append(500*[-2.0])
                                #print(relurl,' rel embedding not found')
                if len(entrels) > 10:
                        continue
                uid = item["uid"]

                start_decoding = vocab.word_to_id(vocab.START_DECODING)
                stop_decoding = vocab.word_to_id(vocab.STOP_DECODING)
                 
                question_words = question.split()[ : max_enc_len]
                fasttext_vectors = []
                for word in question_words:
                    try:
                        fasttext_vectors.append(fasttext.word_vec(word)+200*[0.0])
                    except Exception as err:
                        fasttext_vectors.append(500*[0.0])
                questions_fasttext_vectors = fasttext_vectors + entrelsembeddings
                question_words += entrels [:max_enc_len]
                
                
                #print(question.split(),entrels,question_words)
                enc_len = len(question_words)
                enc_input = [vocab.word_to_id(w) for w in question_words]
                enc_input_extend_vocab, question_oovs = Data_Helper.article_to_ids(question_words, vocab)
                #print("question_wrds: ",question_words)
                #print("enc_input_extend_vocab: ",enc_input_extend_vocab)
                #print("question oovs: ",question_oovs)

                intsparql_words_ = sparql.replace("("," ( ").replace(")"," ) ").replace(","," , ").replace("wd:","").replace("wdt:","").replace("ps:","").replace("pq:","").replace("p:","").split()
                #print("sparql: ",sparql)
                #print("intsparql: ",intsparql_words_)
                intsparql_words = [x.lower() for x in intsparql_words_]
                intsparql_ids = [vocab.word_to_id(w) for w in intsparql_words]
                intsparql_ids_extend_vocab = Data_Helper.abstract_to_ids(intsparql_words, vocab, question_oovs)
                dec_input, target = Data_Helper.get_dec_inp_targ_seqs(intsparql_ids, max_dec_len, start_decoding, stop_decoding)
                #print("decinput:",dec_input)
                #print("tar1: ",target)
                #print("extendvocab: ",intsparql_ids_extend_vocab)
                _, target = Data_Helper.get_dec_inp_targ_seqs(intsparql_ids_extend_vocab, max_dec_len, start_decoding, stop_decoding)
                #print("_: ",_)
                #print("tar2: ",target)
                dec_len = len(dec_input)
         
                output = {
                        "uid":uid,
                        "enc_len":enc_len,
                        "enc_input" : enc_input,
                        "old_enc_input": enc_input,
                        "enc_input_extend_vocab"  : enc_input_extend_vocab,
                        "question_oovs" : question_oovs,
                        "dec_input" : dec_input,
                        "target" : target,
                        "dec_len" : dec_len,
                        "question" : question,
                        "intermediate_sparql" : sparql,
                        "entrels": entrels,
                        "entrelsembeddings": entrelsembeddings,
                        "questions_fasttext_vectors": questions_fasttext_vectors
                }

                yield output

def batch_generator(generator, filenames, vocab_path,  vocab_size, max_enc_len, max_dec_len, batch_size, training):
        dataset = tf.data.Dataset.from_generator(generator, args = [filenames, vocab_path,  vocab_size, max_enc_len, max_dec_len, training],
                                                                                        output_types = {
                                                                                                "uid":tf.int32,
                                                                                                "enc_len":tf.int32,
                                                                                                "enc_input" : tf.int32,
                                                                                                "old_enc_input": tf.int32,
                                                                                                "enc_input_extend_vocab"  : tf.int32,
                                                                                                "question_oovs" : tf.string,
                                                                                                "dec_input" : tf.int32,
                                                                                                "target" : tf.int32,
                                                                                                "dec_len" : tf.int32,
                                                                                                "question" : tf.string,
                                                                                                "intermediate_sparql" : tf.string,
                                                                                                "entrels": tf.string,
                                                                                                "questions_fasttext_vectors": tf.float32,
                                                                                                "entrelsembeddings": tf.float32
                                                                                        }, output_shapes={
                                                                                               "uid": [],
                                                                                                "enc_len":[],
                                                                                                "enc_input" : [None],
                                                                                                "old_enc_input":[None],
                                                                                                "enc_input_extend_vocab"  : [None],
                                                                                                "question_oovs" : [None],
                                                                                                "dec_input" : [None],
                                                                                                "target" : [None],
                                                                                                "dec_len" : [],
                                                                                                "question" : [],
                                                                                                "intermediate_sparql" : [],
                                                                                                "entrels": [None],
                                                                                                "questions_fasttext_vectors": [None,None],
                                                                                                "entrelsembeddings": [None,None]
                                                                                        })
        dataset = dataset.padded_batch(batch_size, padded_shapes=({"enc_len":[],
                                                                                                "enc_input" : [None],
                                                                                                "old_enc_input": [None],
                                                                                                "uid": [],
                                                                                                "enc_input_extend_vocab"  : [None],
                                                                                                "question_oovs" : [None],
                                                                                                "dec_input" : [max_dec_len],
                                                                                                "target" : [max_dec_len],
                                                                                                "dec_len" : [],
                                                                                                "question" : [],
                                                                                                "intermediate_sparql" : [],
                                                                                                "entrels": [None],
                                                                                                "questions_fasttext_vectors": [None,None],
                                                                                                "entrelsembeddings": [None,None]
                                                                                                }),
                                                                                        padding_values={"enc_len":-1,
                                                                                                "uid": -1,
                                                                                                "enc_input" : -1,
                                                                                                "old_enc_input":-1,
                                                                                                "enc_input_extend_vocab"  : 1,
                                                                                                "question_oovs" : b'',
                                                                                                "dec_input" : 1,
                                                                                                "target" : 1,
                                                                                                "dec_len" : -1,
                                                                                                "question" : b"",
                                                                                                "intermediate_sparql" : b"",
                                                                                                "entrels": b"",
                                                                                                "questions_fasttext_vectors": -1.0,
                                                                                                "entrelsembeddings": -1.0},
                                                                                        drop_remainder=True)
        def update(entry):
                return ({"enc_input" : entry["enc_input"],
                        "old_enc_input": entry["old_enc_input"],
                         "uid" : entry["uid"],
                        "extended_enc_input" : entry["enc_input_extend_vocab"],
                        "question_oovs" : entry["question_oovs"],
                        "enc_len" : entry["enc_len"],
                        "question" : entry["question"],
                        "entrels": entry["entrels"],
                        "entrelsembeddings": entry["entrelsembeddings"],
                        "questions_fasttext_vectors": entry["questions_fasttext_vectors"],
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
