import torch

def compute_global_query(conv_queries, dense_queries):
    pooled = []

    for q in conv_queries:
        pooled.append(q.mean(dim=(2, 3)))  

    for q in dense_queries:
        pooled.append(q)  

    q_global = torch.stack(pooled, dim=0).mean(dim=0)  
    return q_global  
