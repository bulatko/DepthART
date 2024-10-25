import torch
from torch import nn




def level_ebmedding(patch_nums, embed_dim, init_std):
    '''
    Create positional and level embeddings
    Input:
        - patch_nums: sizes of each predicted level
        - embeded dim: dim of positional embeding
        - init_std: parameter for initialization
    Returns:
        - lvl_embed: embedding for each level
        - pos_1LC: vecor of parameters for each position
    '''
    pos_1LC = []
    for i, pn in enumerate(patch_nums):
        pe = torch.empty(1, pn*pn, embed_dim)
        nn.init.trunc_normal_(pe, mean=0, std=init_std)
        pos_1LC.append(pe)
    
    pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C

    pos_1LC = nn.Parameter(pos_1LC)
    
    # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
    lvl_embed = nn.Embedding(len(patch_nums), embed_dim)
    nn.init.trunc_normal_(lvl_embed.weight.data, mean=0, std=init_std)
    return lvl_embed, pos_1LC


def class_embedding(num_classes, init_std, embed_dim, first_l):
    '''
    Create class embeddings
    Input:
        - num_classes: number of classes for class-guided generation
        - embed_dim: dim of class embeding (same as for positional)
        - init_std: parameter for initialization
        - first_l: size of first layer
    Returns:
        - class_emb
        - pos_start


    '''

    # create class embedding
    class_emb = nn.Embedding(num_classes + 1, embed_dim)
    nn.init.trunc_normal_(class_emb.weight.data, mean=0, std=init_std)

    # create ????
    pos_start = nn.Parameter(torch.empty(1, first_l, embed_dim))
    nn.init.trunc_normal_(pos_start.data, mean=0, std=init_std)
    
    return class_emb, pos_start


def create_begin_ends(patch_nums):
    begin_ends = []   
    cur = 0
    for i, pn in enumerate(patch_nums):
        begin_ends.append((cur, cur+pn ** 2))
        cur += pn ** 2
    return begin_ends