import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer
import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='webqsp')
args = parser.parse_args()

dataset=args.dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if dataset=='webqsp':
    relation_file = 'data/webqsp/relations.txt'
    entity_file = 'data/webqsp/entities.txt'
    entities_names_file = 'entities_names_all.json'
    relation_embedding_file = 'data/webqsp/relation_embeddings.npy'
    entity_embedding_file = 'data/webqsp/entity_embeddings_all.npy'

elif dataset=='CWQ':
    relation_file = 'data/CWQ/relations.txt'
    entity_file = 'data/CWQ/entities.txt'
    entities_names_file = 'entities_names_all2.json'
    relation_embedding_file = 'data/CWQ/relation_embeddings.npy'
    entity_embedding_file = 'data/CWQ/entity_embeddings_all.npy'

model_name = 'distilbert-base-uncased' 
embedding_dim = 768
reduced_dim = 128
batch_size = 16

tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name).to(device)
model.eval()

with open(relation_file, 'r') as f:
    relations = [line.strip() for line in f.readlines()]

with open(entity_file, 'r') as f:
    entity_ids = [line.strip() for line in f.readlines()]

with open(entities_names_file, 'r') as f:
    entities_names = json.load(f)

filtered_entities = {eid: entities_names[eid] for eid in entity_ids if eid in entities_names}
filtered_entities = {eid: entities_names.get(eid, eid) for eid in entity_ids}

def prepare_text_batches(relations, entities):
    texts = relations + list(entities.values())
    return [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]


def compute_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state

    
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    valid_token_sum = (last_hidden_state * mask_expanded).sum(dim=1)
    valid_token_count = mask_expanded.sum(dim=1)
    
    
    valid_token_count = valid_token_count.clamp(min=1e-9)
    embeddings = valid_token_sum / valid_token_count

    return embeddings.cpu().numpy()


def reduce_dimensions(embeddings, dim):
    pca = PCA(n_components=dim)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

relation_batches = prepare_text_batches(relations, {})
entity_batches = prepare_text_batches([], filtered_entities)

relation_embeddings = []
entity_embeddings = []

print("Computing embeddings for relations...")
for batch in tqdm(relation_batches):
    relation_embeddings.append(compute_embeddings(batch))

print("Computing embeddings for entities...")
for batch in tqdm(entity_batches):
    entity_embeddings.append(compute_embeddings(batch))

relation_embeddings = np.vstack(relation_embeddings)
entity_embeddings = np.vstack(entity_embeddings)


print("Reducing dimensions...")
relation_embeddings = reduce_dimensions(relation_embeddings, reduced_dim)
entity_embeddings = reduce_dimensions(entity_embeddings, reduced_dim)

np.save(relation_embedding_file, relation_embeddings)
np.save(entity_embedding_file, entity_embeddings)

print("Embeddings saved.")
