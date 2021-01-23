import sys,json, torch
from elasticsearch import Elasticsearch
from transformers import BertTokenizer, BertModel

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
model.to('cuda')
d = json.loads(open('wikidataprops.json').read())
wikidatapropembeds = []
for idx,item in enumerate(d):
    print(item)
    desc = item['description']
    id = item['id']
    if not desc:
        continue
    text = '[CLS] '+desc+ ' [SEP]'
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda')
    with torch.no_grad():
        outputs = model(tokens_tensor)
        encoded_layers = outputs[0]
        embedding = encoded_layers[0][0]
        wikidatapropembeds.append({'id':id,'desc':desc,'embedding':[str(x) for x in list(embedding.cpu().numpy())]})
        print("idx :",idx," id: ",id, " desc: ",desc," tokens: ",tokenized_text)
        #print("enc layer: ",encoded_layers.shape)
        #print("wikidataar : ",json.dumps(wikidatapropembeds))

f = open('wikidatapropembeddings.json','w')
f.write(json.dumps(wikidatapropembeds))
f.close()
