
import time
import numpy as np
import os, math

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertModel, DistilBertTokenizer
from tqdm import tqdm
import anthropic
import json
import openai
from prompt_list import *
import re
from dataset_load import load_data
import random
openai_key="<your_key>"
claude_client = anthropic.Anthropic(
    api_key="<your_key>",
)
class QuestionAwareGNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers):
        """
        Initialize the GNN model.
        :param embedding_dim: Dimension of the input embeddings (for nodes, relations, and questions).
        :param hidden_dim: Dimension of the hidden layers.
        :param num_layers: Number of GNN layers.
        """
        super(QuestionAwareGNN, self).__init__()
        
        self.num_layers = num_layers
        
        self.edge_update_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4 * embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim),
            ) for _ in range(num_layers)
        ])
        self.node_update_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2* embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
            for _ in range(num_layers)
        ])
        
        self.edge_score_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        self.node_score_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        self.question_projection = nn.Linear(768, embedding_dim)
    
    def forward(self, node_embedding, relation_embedding, question_embedding, edge_index, edge_type):
        """
        Forward pass of the GNN.
        :param node_embedding: Node embeddings for the subgraph (num_nodes, embedding_dim).
        :param relation_embedding: Relation embeddings for the subgraph (num_edges, embedding_dim).
        :param question_embedding: Encoded question embedding (1, embedding_dim).
        :param edge_index: Edge index of the subgraph (2, num_edges).
        :param edge_type: Edge types of the subgraph (num_edges).
        :return: Node scores, edge scores.
        """
        
        node_hidden = node_embedding
        edge_hidden = relation_embedding
        question_embedding = self.question_projection(question_embedding)
        
        for layer in range(self.num_layers):
            edge_source = node_hidden[edge_index[0]]
            edge_target = node_hidden[edge_index[1]]
            edge_features = torch.cat(
                [edge_source, edge_target, edge_hidden, question_embedding.expand(edge_source.size(0), -1)], dim=-1
            )
            updated_edge_hidden = self.edge_update_mlp[layer](edge_features)
            
            aggregated_messages = torch.zeros_like(node_hidden)
            aggregated_messages.index_add_(0, edge_index[0], updated_edge_hidden)
            aggregated_messages.index_add_(0, edge_index[1], updated_edge_hidden)

            
            updated_node_hidden = self.node_update_mlp[layer](
                torch.cat([node_hidden, aggregated_messages], dim=-1)
            )
            
            node_hidden = updated_node_hidden
            edge_hidden = updated_edge_hidden

        edge_scores = self.edge_score_mlp(edge_hidden).squeeze(-1)
        node_scores = self.node_score_mlp(node_hidden).squeeze(-1)
        
        return node_scores, edge_scores
    



class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.load_data(args)
        


    def load_data(self, args):
        dataset = load_data(args)
        self.train_data = dataset["train"]
        self.valid_data = dataset["valid"]
        self.test_data = dataset["test"]
        self.entity2id = dataset["entity2id"]
        self.relation2id = dataset["relation2id"]

        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        

    def transfer_edge(self, kb_adj_mats_list):
        edge_index_list=[]
        edge_type_list=[]
        for kb_adj_mats in kb_adj_mats_list:
            head_list, rel_list, tail_list = kb_adj_mats
            edge_index = torch.tensor([head_list, tail_list], dtype=torch.long).to(self.device)
            edge_type = torch.tensor(rel_list, dtype=torch.long).to(self.device)

            edge_index_list.append(edge_index)
            edge_type_list.append(edge_type)
        
        return edge_index_list, edge_type_list

    def extract_subgraph_embeddings(self,local_entity, global_node_embedding, global_relation_embedding, edge_index, edge_type, g2l):
        """
        Extract embeddings for nodes and relations specific to the subgraph.
        :param global_node_embedding: Global node embeddings (num_global_nodes, embedding_dim).
        :param global_relation_embedding: Global relation embeddings (num_global_relations, embedding_dim).
        :param edge_index: Edge index of the subgraph (2, num_edges).
        :param edge_type: Edge types of the subgraph (num_edges).
        :param g2l: Global-to-local ID mapping (dict).
        :return: local_node_embedding, local_relation_embedding, local_to_global
        """
        # Find all unique nodes in the subgraph
        l2g = {l: g for g, l in g2l.items()}
        
        edge_nodes = torch.unique(edge_index.flatten())#.tolist()  # Nodes from edge_index
        local_node_ids = torch.unique(
            torch.cat([torch.tensor(local_entity, dtype=torch.int).to(self.device), edge_nodes])
        ).tolist()
        
        global_node_ids = [l2g[n] for n in local_node_ids]
        
        local_node_embedding = global_node_embedding[global_node_ids]
        local_relation_embedding = global_relation_embedding[edge_type]
        return local_node_embedding, local_relation_embedding, l2g



    def find_paths(self, edge_index, source_entities, answer_local, max_hops):
        """
        Finds all nodes and edges that belong to valid paths from source_entities to answer_local within max_hops.

        Args:
            edge_index (torch.Tensor): [2, E] tensor representing edges in the graph.
            source_entities (set): Set of source node indices.
            answer_local (set): Set of target node indices.
            max_hops (int): Maximum allowable path length.

        Returns:
            positive_nodes (set): Set of node indices appearing on valid paths.
            positive_edges (set): Set of edge indices (edge_index column indices) appearing on valid paths.
        """
        # Initialize result containers
        positive_nodes = set()
        positive_edges = set()

        adjacency_list = {}
        edge_to_index = {}
        for idx, (src, dst) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
            adjacency_list.setdefault(src, []).append((dst, idx))
            adjacency_list.setdefault(dst, []).append((src, idx))
            edge_to_index[(min(src, dst), max(src, dst))] = idx

        def dfs(current_node, path_nodes, path_edges, depth):
            """
            Recursive DFS function to explore all paths within max_hops.
            """
            if depth > max_hops:
                return False

            if current_node in answer_local:
                positive_nodes.update(path_nodes)
                positive_edges.update(path_edges)
                return True

            found_valid_path = False

            for neighbor, edge_idx in adjacency_list.get(current_node, []):
                if edge_idx not in path_edges:
                    if dfs(
                        neighbor, 
                        path_nodes | {neighbor},
                        path_edges | {edge_idx},
                        depth + 1
                    ):
                        found_valid_path = True

            return found_valid_path

        # Perform DFS for each source entity
        for source in source_entities:
            dfs(source, {source}, set(), 0)

        return positive_nodes, positive_edges


    def get_edge(self, data, max_hops=3,dataset='train'):
        device=self.device
        
        num_data=data.num_data
        g2l_list=data.global2local_entity_maps
        kb_adj_mats_list=data.kb_adj_mats
        
        edge_index_list,edge_type_list=self.transfer_edge(kb_adj_mats_list)
        answer_lists=data.answer_lists
        
        for i in tqdm(range(num_data)):

            g2l=g2l_list[i]
            global_entity=data.data[i]['entities']
            local_entity=[]
            for g_e in global_entity:
                local_entity.append(g2l[g_e])

            kb_adj_mats=kb_adj_mats_list[i]
            edge_index=edge_index_list[i]
            edge_type=edge_type_list[i]
            answers_id=answer_lists[i]
            answer_local = set(g2l[ans] for ans in answers_id if ans in g2l)
            source_entities = set(g2l[ent] for ent in global_entity if ent in g2l)
            try:
                positive_nodes, positive_edges = self.find_paths(edge_index, source_entities, answer_local, max_hops)
                
                
                new_entry={'id':i,'positive_nodes':positive_nodes,'positive_edges':positive_edges}
                with open(f'data/CWQ/edge_{dataset}.json', "a", encoding="utf-8") as f:
                    new_entry = {key: list(value) if isinstance(value, set) else value for key, value in new_entry.items()}
                    f.write(json.dumps(new_entry) + '\n')
            except:
                continue

    def get_pos(self,i,data_set='train'):
        if data_set=='train':
            if i in self.p_train:
                return self.p_train[i]['positive_nodes'], self.p_train[i]['positive_edges']
            else:
                return set(), set()
        elif data_set=='dev':
            if i in self.p_dev:
                return self.p_dev[i]['positive_nodes'], self.p_dev[i]['positive_edges']
            else:
                return set(), set()
    def get_p(self,file):
        result_dict = {}
        with open(file, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    result_dict[data["id"]] = {
                    "positive_nodes": set(data["positive_nodes"]),
                    "positive_edges": set(data["positive_edges"])
                    }
                except:
                    continue
                
        print(len(result_dict))
        return result_dict
        
    
    def train_gnn(self, data, data2, max_hops=3, epochs=50):

        saved_edge=False
        if saved_edge:
            file_train="data/CWQ/edge_train.json"
            file_dev="data/CWQ/edge_dev.json"
            self.p_train=self.get_p(file_train)
            self.p_dev=self.get_p(file_dev)
            
        device=self.device
        
        num_data=data.num_data
        g2l_list=data.global2local_entity_maps
        kb_adj_mats_list=data.kb_adj_mats
        
        edge_index_list,edge_type_list=self.transfer_edge(kb_adj_mats_list)
        answer_lists=data.answer_lists
        
        
        num_data2=data2.num_data
        g2l_list2=data2.global2local_entity_maps
        candidate_entities_list2 =data2.candidate_entities 
        kb_adj_mats_list2=data2.kb_adj_mats
        
        edge_index_list2,edge_type_list2=self.transfer_edge(kb_adj_mats_list2)
        query_entities_list2=data2.query_entities
        answer_dists2=data2.answer_dists
        answer_lists2=data2.answer_lists
        
        self.gnn.train()
        optimizer = torch.optim.Adam(self.gnn.parameters(), lr=1e-4)
        optimizer.zero_grad()
        
        best_loss=100
        for ep in (range(epochs)):
            skip_wrong_index=0
            for i in tqdm(range(num_data)):
                optimizer.zero_grad()
                
                question=data.data[i]['question']
                with torch.no_grad():
                    inputs =  self.tokenizer(question, padding=True, truncation=True, return_tensors="pt").to(device)
                    outputs =  self.model(**inputs)
                    question_embedding = outputs.last_hidden_state.mean(dim=1).to(device)

                g2l=g2l_list[i]

                global_entity=data.data[i]['entities']
                local_entity=[]
                for g_e in global_entity:
                    local_entity.append(g2l[g_e])

                kb_adj_mats=kb_adj_mats_list[i]
                edge_index=edge_index_list[i]
                edge_type=edge_type_list[i]
                answers_id=answer_lists[i]
                
                with torch.no_grad():
                    node_embedding, relation_embedding, local_to_global=self.extract_subgraph_embeddings(local_entity, self.entity_embeddings, self.relation_embedding, edge_index, edge_type, g2l)
                if node_embedding.shape[0]<=torch.max(edge_index[0]) or node_embedding.shape[0]<=torch.max(edge_index[1]):
                    skip_wrong_index+=1
                    continue
                answer_local = set(g2l[ans] for ans in answers_id if ans in g2l)
                
                source_entities = set(g2l[ent] for ent in global_entity if ent in g2l)

                
                node_scores, edge_scores = self.gnn(
                    node_embedding, relation_embedding, question_embedding, edge_index, edge_type
                )
                
                try:
                    if not saved_edge:
                        positive_nodes, positive_edges = self.find_paths(edge_index, source_entities, answer_local, max_hops)
                    else:
                        positive_nodes, positive_edges = self.get_pos(i,'train')
                except:
                    optimizer.zero_grad()
                    continue
                
                try:
                    all_nodes = set(range(node_embedding.size(0)))
                    all_edges = set(range(edge_index.shape[1]))
                    negative_nodes = all_nodes - set(positive_nodes)
                    negative_edges = all_edges - set(positive_edges)
                    if len(positive_nodes) > 0 and len(positive_edges)>0:
                        pos_node_loss = -torch.log(node_scores[list(positive_nodes)] + 1e-9).mean()
                        neg_node_loss = -torch.log(1 - node_scores[list(negative_nodes)] + 1e-9).mean()

                        pos_edge_loss = -torch.log(edge_scores[list(positive_edges)] + 1e-9).mean()
                        neg_edge_loss = -torch.log(1 - edge_scores[list(negative_edges)] + 1e-9).mean()
                    else:
                        pos_node_loss = 0
                        neg_node_loss = -torch.log(1 - node_scores[list(negative_nodes)] + 1e-9).mean()

                        pos_edge_loss = 0
                        neg_edge_loss = -torch.log(1 - edge_scores[list(negative_edges)] + 1e-9).mean()

                    loss = pos_node_loss + neg_node_loss + pos_edge_loss + neg_edge_loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                except:
                    optimizer.zero_grad()
                    continue
            print('skip:', skip_wrong_index)
            loss_dev=0
            
            for i in tqdm(range(num_data2)):
    
                question=data2.data[i]['question']
                
                with torch.no_grad():
                    inputs =  self.tokenizer(question, padding=True, truncation=True, return_tensors="pt").to(device)
                    outputs =  self.model(**inputs)
                    question_embedding = outputs.last_hidden_state.mean(dim=1).to(device)
                
                g2l=g2l_list2[i]

                global_entity=data2.data[i]['entities']
                local_entity=[]
                for g_e in global_entity:
                    local_entity.append(g2l[g_e])

                kb_adj_mats=kb_adj_mats_list2[i]
                edge_index=edge_index_list2[i]
                edge_type=edge_type_list2[i]
                answers_id=answer_lists2[i]
                
                with torch.no_grad():
                    node_embedding, relation_embedding, local_to_global=self.extract_subgraph_embeddings(local_entity, self.entity_embeddings, self.relation_embedding, edge_index, edge_type, g2l)
                if node_embedding.shape[0]<=torch.max(edge_index[0]) or node_embedding.shape[0]<=torch.max(edge_index[1]):
                    continue
                answer_local = set(g2l[ans] for ans in answers_id if ans in g2l)
                source_entities = set(g2l[ent] for ent in global_entity if ent in g2l)

                with torch.no_grad():
                    node_scores, edge_scores = self.gnn(
                        node_embedding, relation_embedding, question_embedding, edge_index, edge_type
                    )
                
                try:
                    if not saved_edge:
                        positive_nodes, positive_edges = self.find_paths(edge_index, source_entities, answer_local, max_hops)
                    else:
                        positive_nodes, positive_edges = self.get_pos(i,'dev')
                except:
                    continue
                
                try:
                    all_nodes = set(range(node_embedding.size(0)))
                    all_edges = set(range(edge_index.shape[1]))
                    negative_nodes = all_nodes - set(positive_nodes)
                    negative_edges = all_edges - set(positive_edges)
                    if len(positive_nodes) > 0 and len(positive_edges)>0:
                        pos_node_loss = -torch.log(node_scores[list(positive_nodes)] + 1e-9).mean()
                        neg_node_loss = -torch.log(1 - node_scores[list(negative_nodes)] + 1e-9).mean()

                        pos_edge_loss = -torch.log(edge_scores[list(positive_edges)] + 1e-9).mean()
                        neg_edge_loss = -torch.log(1 - edge_scores[list(negative_edges)] + 1e-9).mean()
                    else:
                        pos_node_loss = 0
                        neg_node_loss = -torch.log(1 - node_scores[list(negative_nodes)] + 1e-9).mean()

                        pos_edge_loss = 0
                        neg_edge_loss = -torch.log(1 - edge_scores[list(negative_edges)] + 1e-9).mean()

                    loss = pos_node_loss + neg_node_loss + pos_edge_loss + neg_edge_loss
                    if loss>100:
                        loss=100
                    loss_dev += loss
                except:
                    continue
            loss_dev/=num_data2
            print('The dev loss: ',loss_dev)
            if loss_dev<best_loss:
                best_loss=loss_dev
                torch.save(self.gnn.state_dict(), f'data/{self.dataset}/gnn.pth')                
    
    def e2n(self,entity):
        if entity in self.entities_names:
            return self.entities_names[entity]
        else:
            return entity
        
    def ggr(self, eval=True):
        
        use_class=True
        plm=True
        use_llama=False
        show_edge=False
        show_info=False
        #self.llm_name='gpt3.5'
        #self.llm_name='gpt4'
        self.llm_name='Claude'
        #self.dataset='CWQ'
        self.dataset='webqsp'
        device=self.device
        if self.dataset=='webqsp':
            e2n_file='entities_names_all.json'
        elif self.dataset=='CWQ':
            e2n_file='entities_names_all2.json'
        with open(e2n_file) as f:
            entities_names = json.load(f)#m.06w2sn5 Justin Bieber
            self.entities_names=entities_names
        
        names_entities = {v: k for k, v in entities_names.items()}
        self.names_entities=names_entities
        
        if eval==False:
            data=self.train_data
            data2=self.valid_data
        else:
            data=self.test_data
        
        num_data=data.num_data
        g2l_list=data.global2local_entity_maps
        kb_adj_mats_list=data.kb_adj_mats
        
        edge_index_list,edge_type_list=self.transfer_edge(kb_adj_mats_list)
        
        query_entities_list=data.query_entities
        answer_dists=data.answer_dists
        answer_lists=data.answer_lists
        
        max_hops=2
        self.gnn=QuestionAwareGNN(embedding_dim=128, hidden_dim=512*4, num_layers=3).to(device)
        
        if eval==True:
            state_dict = torch.load(f'data/{self.dataset}/gnn.pth') 
            self.gnn.load_state_dict(state_dict)   
        
        
        
        entity_embeddings = np.load(f'data/{self.dataset}/entity_embeddings_all.npy')
        self.entity_embeddings = torch.FloatTensor(entity_embeddings).to(device)
        
        relation_embedding = np.load(f'data/{self.dataset}/relation_embeddings.npy')
        self.relation_embedding = torch.FloatTensor(relation_embedding).to(device)
        if plm:
            model_name = 'distilbert-base-uncased'
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model = DistilBertModel.from_pretrained(model_name).to(device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
                
        
        if eval==False:
            self.train_gnn(data, data2, max_hops=2, epochs=50)
            return 0
        
        
        save_file=f'data/{self.dataset}/llm_answer_class_0.9_0.1_{self.llm_name}_2hop.json'
        skip_wrong_index=0
        correct=0
        for i in tqdm(range(num_data)):
            
            question=data.data[i]['question']
            
            with torch.no_grad():
                inputs =  self.tokenizer(question, padding=True, truncation=True, return_tensors="pt").to(device)
                outputs =  self.model(**inputs)
                question_embedding = outputs.last_hidden_state.mean(dim=1).to(device)
            
            g2l=g2l_list[i]
            
            global_entity=data.data[i]['entities']
            local_entity=[]
            
            kb_adj_mats=kb_adj_mats_list[i]
            edge_index=edge_index_list[i]
            edge_type=edge_type_list[i]

            query_entities=query_entities_list[i]
            answer_ones=answer_dists[i]
            answers_id=answer_lists[i]
            
            
            
            
            if show_edge:
                answer_local = set(g2l[ans] for ans in answers_id if ans in g2l)
                source_entities = set(g2l[ent] for ent in global_entity if ent in g2l)
                positive_nodes, positive_edges = self.find_paths(edge_index, source_entities, answer_local, max_hops=2)

                print('positive_edges',positive_edges)
                print('positive_nodes',positive_nodes)
            
            head_list, rel_list, tail_list = kb_adj_mats
            local_to_global = {v: k for k, v in g2l.items()} 
            
            
            node_embedding, relation_embedding, local_to_global=self.extract_subgraph_embeddings(local_entity, self.entity_embeddings, self.relation_embedding, edge_index, edge_type, g2l)
            
            if edge_index[0].numel() > 0:
                if node_embedding.shape[0]<=torch.max(edge_index[0]) or node_embedding.shape[0]<=torch.max(edge_index[1]):
                    skip_wrong_index+=1
                    print('skip:', skip_wrong_index)
                    updated_answer = {
                    "id": i,
                    "question": question,
                    "answer": 'error'
                    }
                    with open(save_file, 'a') as f:
                        f.write(json.dumps(updated_answer) + '\n')
                    continue

            with torch.no_grad():
                graph_node_score, graph_edge_score=self.gnn(node_embedding, relation_embedding, question_embedding, edge_index, edge_type)
            
            
            if show_edge:
                print(graph_node_score[:20])
                print(graph_edge_score[:20])
                print(graph_node_score[list(positive_nodes)])
                print(graph_edge_score[list(positive_edges)])
                print('-'*200)
                #if i>5:
                #    exit()
                continue
            #exit()
            paths = {entity: [(entity,)] for entity in global_entity}
            helpful_triplets = []
            
            max_hops=2
            if self.dataset=='CWQ':
                max_hops=2
            #print('paths:',paths)
            for hop in range(max_hops):
                new_paths = {}
                selected_relations = set()  # To store relations selected by LLM

                # Step 1: Find related relations for current entities
                #print('paths:',paths)
                #print('hop ',hop+1,'step 1')
                for entity, current_paths in paths.items():
                    
                    collected_values = set()
                    if hop>0:
                        for c_path in current_paths:
                            for path in c_path:
                                if isinstance(path, tuple):
                                    collected_values.add(g2l[path[0]])  
                                    collected_values.add(g2l[path[2]])
                    #print(' '*10,'collected_values: ',collected_values)
                    #print(' '*10,'entity:',entity)
                    entity_local_id = g2l[entity]
                    connected_edges = [
                        (head_list[j], rel_list[j], tail_list[j], round(graph_edge_score[j].item(),3))
                        for j in range(len(head_list))
                        if (head_list[j] == entity_local_id and tail_list[j] not in collected_values) or (tail_list[j] == entity_local_id and head_list[j] not in collected_values)
                    ]
                    #print(connected_edges)
                    #Input for LLM: Relation candidates with scores
                    relation_candidates = {}
                    for head, rel, tail, edge_score in connected_edges:
                        relation_candidates[self.id2relation[rel]] = max(relation_candidates.get(self.id2relation[rel], 0), edge_score)

                    #print(' '*10,'entity:',entity,'relation_candidates:',len(relation_candidates))
                    
                    
                    if use_class:
                        relation_candidates_gnn={}
                        relation_candidates_update={}
                        for rel, score in relation_candidates.items():
                            if score>0.9:
                                relation_candidates_gnn[rel]=score
                                
                            elif score<=0.9 and score>0.1:
                                with torch.no_grad():
                                    inputs_rel =  self.tokenizer(rel, padding=True, truncation=True, return_tensors="pt").to(device)
                                    outputs_rel =  self.model(**inputs_rel)
                                    rel_embedding = outputs_rel.last_hidden_state.mean(dim=1).to(device)
                                    relation_candidates_update[rel]=torch.nn.functional.cosine_similarity(question_embedding,rel_embedding).item()
                        if len(relation_candidates_gnn)>5:
                            sorted_dict=sorted(relation_candidates_gnn.items(), key=lambda item: item[1], reverse=True)
                            selected_gnn_cand = dict(sorted_dict[:5])
                            unselected_gnn_cand = dict(sorted_dict[5:])
                            for rel, score in unselected_gnn_cand.items():
                                with torch.no_grad():
                                    inputs_rel =  self.tokenizer(rel, padding=True, truncation=True, return_tensors="pt").to(device)
                                    outputs_rel =  self.model(**inputs_rel)
                                    rel_embedding = outputs_rel.last_hidden_state.mean(dim=1).to(device)
                                    relation_candidates_update[rel]=torch.nn.functional.cosine_similarity(question_embedding,rel_embedding).item()
                        else:
                            selected_gnn_cand=relation_candidates_gnn
                        
                        num_gnn=0
                        for rel, score in selected_gnn_cand.items():
                            if rel in self.relation2id:
                                rel = self.relation2id[rel]
                                selected_relations.add(rel)
                                num_gnn+=1
                                #print(' '*20,'selected_rel: ',rel)
                        #print(' '*20,'selected_gnn_cand:',num_gnn)        
                        
                        relation_candidates=relation_candidates_update
                    
                    
                    if len(relation_candidates)==0:
                        continue
                    
                    # Prepare text for LLM
                    relations_text = " ".join(
                        [f"{rel}({score});" for rel, score in relation_candidates.items()]
                    )
                    #print(' '*30,'relations_text: ',relations_text)
                    node_text = self.e2n(self.id2entity[entity])
                    if use_class:
                        llm_input_relations = f"{extract_relation_prompt_encode}{question}\nTopic Entity: {node_text}\nRelations: {relations_text}\nA: "
                    else:
                        llm_input_relations = f"{extract_relation_prompt}{question}\nTopic Entity: {node_text}\nRelations: {relations_text}\nA: "
                    # Call LLM to select helpful relations
                    
                    #print(llm_input_relations)
                    #print('-'*100)
                    llm_output_relations = self.get_response(llm_input_relations).split("Q: ")[0]
                    
                    #print(' '*30,'llm_output: ',llm_output_relations)
                    #print('-'*100)
                    pattern_relations = r"\{(.*?)\}"
                    matches = re.findall(pattern_relations, llm_output_relations)
                    num_mid=0
                    for match in matches:
                        if num_mid>=3:
                            continue
                        if match in self.relation2id:
                            rel = self.relation2id[match]
                            selected_relations.add(rel)
                            num_mid+=1
                
                    
                # Step 2: Expand paths based on selected relations and find next entities
                #print('hop ',hop+1,'step 2')
                for entity, current_paths in paths.items():
                    
                    collected_values = set()
                    if hop>0:
                        for c_path in current_paths:
                            for path in c_path:
                                if isinstance(path, tuple):
                                    collected_values.add(g2l[path[0]])  
                                    collected_values.add(g2l[path[2]]) 
                        
                    entity_local_id = g2l[entity]

                    # Gather all edges connected to the current entity
                    connected_edges = [
                        (head_list[j], rel_list[j], tail_list[j], graph_edge_score[j])
                        for j in range(len(head_list))
                        if (head_list[j] == entity_local_id and tail_list[j] not in collected_values) or (tail_list[j] == entity_local_id and head_list[j] not in collected_values)
                    ]

                    # Iterate over each selected relation
                    for selected_relation in selected_relations:
                        #print(' '*10,'selected_relation: ',selected_relation)
                        # Filter edges that match the current relation
                        valid_edges = [
                            (head, rel, tail, edge_score)
                            for head, rel, tail, edge_score in connected_edges
                            if rel == selected_relation
                        ]
                        if not use_class:
                            if len(valid_edges)>20:
                                valid_edges = random.sample(valid_edges, 20)
                        if len(valid_edges)<=1:
                            #print(' '*20,'only 1 edge')
                            for head, rel, tail, edge_score in valid_edges:
                                if head == entity_local_id:
                                    next_entity_local = tail
                                elif tail == entity_local_id:
                                    next_entity_local = head
                                else:
                                    continue
                                next_entity_global = local_to_global[next_entity_local]
                                if next_entity_global not in self.id2entity:
                                    continue
                                #print(' '*20,'next_entity_global: ',next_entity_global)
                                # Add new paths
                                for path in current_paths:
                                    new_path = path + ((entity, selected_relation, next_entity_global),)
                                    new_paths.setdefault(next_entity_global, []).append(new_path)
                                    #print(' '*90,'new_paths: ',new_paths)
                                # Record the triplet
                                helpful_triplets.append((entity, selected_relation, next_entity_global))
                            continue
                        # Prepare text for LLM: include only new entities
                        candidate_entities = {}
                        for head, rel, tail, edge_score in valid_edges:
                            
                            if head == entity_local_id:
                                next_entity_local = tail
                            elif tail == entity_local_id:
                                next_entity_local = head
                            else:
                                continue
                            
                            next_entity_global = local_to_global[next_entity_local]
                            if next_entity_global not in self.id2entity:
                                continue
                            #print(' '*60,'next_entity_global',next_entity_global)
                            entity_name=self.e2n(self.id2entity[next_entity_global])
                            candidate_entities[entity_name] = round(graph_node_score[next_entity_local].item(),3)

                        #print(' '*20,'selected_entities: ',len(candidate_entities))
                        
                        if use_class:
                            candidate_entities_gnn={}
                            candidate_entities_update={}
                            for ent, score in candidate_entities.items():
                                if score>0.9:
                                    candidate_entities_gnn[ent]=score
                                elif score<=0.9 and score>0.1:
                                    candidate_entities_update[ent]=score
                            if len(candidate_entities_gnn)>5:
                                sorted_dict=sorted(candidate_entities_gnn.items(), key=lambda item: item[1], reverse=True)
                                selected_gnn_cand = dict(sorted_dict[:5])
                                unselected_gnn_cand = dict(sorted_dict[5:])
                                for ent, score in unselected_gnn_cand.items():
                                    candidate_entities_update[ent]=score
                            else:
                                selected_gnn_cand=candidate_entities_gnn
                                
                            num_gnn=0
                            for ent, score in selected_gnn_cand.items():
                                if ent in names_entities:
                                    if names_entities[ent] in self.entity2id:
                                        next_entity_global = self.entity2id[names_entities[ent]]
                                elif ent in self.entity2id:
                                    next_entity_global = self.entity2id[ent]
                                if next_entity_global not in g2l:
                                    continue
                                next_entity_local = g2l[next_entity_global]
                                #print(' '*20,'next_entity_global: ',next_entity_global)
                                # Add new paths
                                for path in current_paths:
                                    new_path = path + ((entity, selected_relation, next_entity_global),)
                                    new_paths.setdefault(next_entity_global, []).append(new_path)
                                    #print(' '*90,'new_paths: ',new_paths)
                                # Record the triplet
                                helpful_triplets.append((entity, selected_relation, next_entity_global))
                                num_gnn+=1
                            #print(' '*20,'selected_gnn_cand:',num_gnn)   
                            
                            candidate_entities=candidate_entities_update
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        if len(candidate_entities)==0:
                            continue
                        
                        # Prepare input for LLM
                        entities_text = " ".join(
                            [f"{entity}({score});" for entity, score in candidate_entities.items()]
                        )
                        relation_text=self.id2relation[selected_relation]
                        llm_input_entities = f"{entity_candidates_prompt}{question}\nRelation: {relation_text}\nEntites: {entities_text}\nA: "

                        # Call LLM to select next helpful entities
                        #print(' '*30,'entities_text: ',entities_text)
                        llm_output_entities = self.get_response(llm_input_entities).split("Q: ")[0]
                        #print(' '*30,'llm_output_entities: ',llm_output_entities)
                        pattern_entities = r"\{(.*?)\}"

                        # Parse LLM output to get selected entities
                        matches = re.findall(pattern_entities, llm_output_entities)
                        num_ent=0
                        for match in matches:
                            if num_ent>=3:
                                continue
                            if match in names_entities:
                                if names_entities[match] in self.entity2id:
                                    next_entity_global = self.entity2id[names_entities[match]]
                            elif match in self.entity2id:
                                next_entity_global = self.entity2id[match]
                            if next_entity_global not in g2l:
                                continue
                            next_entity_local = g2l[next_entity_global]
                            #print(' '*20,'next_entity_global: ',next_entity_global)
                            # Add new paths
                            for path in current_paths:
                                new_path = path + ((entity, selected_relation, next_entity_global),)
                                new_paths.setdefault(next_entity_global, []).append(new_path)
                                #print(' '*90,'new_paths: ',new_paths)
                            # Record the triplet
                            helpful_triplets.append((entity, selected_relation, next_entity_global))
                            num_ent+=1
                        #print(' '*20,'selected_ent:',num_ent)
                
                # Update for the next hop
                paths = new_paths
                #print(' '*10,'paths: ',paths)
                #print('-'*100)
                #print(' '*10,'helpful_triplets: ',helpful_triplets)
                #print('-'*100)
                # Stop if no new paths
                #exit()
                if not paths:
                    break
                
            #print('helpful_triplets: ',helpful_triplets)
            evidence_text = "\n".join(
                            [f"{self.e2n(self.id2entity[triplet[0]])}, {self.id2relation[triplet[1]]}, {self.e2n(self.id2entity[triplet[2]])};" for triplet in helpful_triplets]
                        )
            llm_input_answers=f"{answer_prompt2}\nQ: {question}\nKnowledge Triplets: {evidence_text}\nA: "
            llm_output_answers = self.get_response(llm_input_answers)
            updated_answer = {
                "id": i,
                "question": question,
                "evidence_text": evidence_text,
                "answer": llm_output_answers
                }
            
            
            with open(save_file, 'a') as f:
                f.write(json.dumps(updated_answer) + '\n')
            flag=0
            for answer_id in answer_lists[i]:
                if answer_id in self.id2entity:
                    if self.e2n(self.id2entity[answer_id]).lower() in llm_output_answers.lower():
                        correct+=1
                        flag=1
                        break
            if flag==0 and True:
                    answer_texts=" ".join(
                        [f"{self.e2n(self.id2entity[answer_id]).lower()}," for answer_id in answer_lists[i] if answer_id in self.id2entity]
                    )
                    input_check=f"Please check if any one of the given answers is provided in the text. Your answer should be yes or no. If any one is mentioned, give 'yes'. Answers: {answer_texts} Text: {llm_output_answers}"
                    output_check=self.get_response(input_check)
                    if 'yes' in output_check.lower():
                        correct+=1
            
            
        print('total correct answers: ',correct,'/',num_data,'=',correct/num_data)    
            
            
            
            
            
            
            
    def chatgpt_response(self, api_key, input_text):
        """Generate a response using ChatGPT/GPT-4 API."""
        if self.llm_name=='gpt3.5':
            model="gpt-3.5-turbo"
        elif self.llm_name=='gpt4':
            model="gpt-4o-mini"
        openai.api_key = api_key
        f=0
        try_times=0
        while(f == 0 and try_times<=10):
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": input_text}
                    ]
                )
                f = 1
            except:
                try_times+=1
                print("error, retry")
                time.sleep(2)
                if try_times>10:
                    return 'error'
        return response['choices'][0]['message']['content']
    
    def Claude_response(self, input_text):
        f=0
        try_times=0
        while(f == 0 and try_times<=10):
            try:
                message = claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": input_text}
                ]
                )
                f = 1
            except:
                try_times+=1
                print("error, retry")
                time.sleep(2)
                if try_times>10:
                    return 'error'
        return message.content[0].text

    def get_response(self, input_text, source="chatgpt", openai_api_key=openai_key):
        """Get response from LLMs."""
        if self.llm_name=='Claude':
            return self.Claude_response(input_text)
        
        elif self.llm_name=='gpt3.5' or self.llm_name=='gpt4':
            if not openai_api_key:
                raise ValueError("OpenAI API key must be provided.")
            return self.chatgpt_response(openai_api_key, input_text)
        else:
            raise ValueError("Unsupported source. Choose 'llama' or 'chatgpt'.")

    