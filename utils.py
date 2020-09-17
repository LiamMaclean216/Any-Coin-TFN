import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2,1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
    return torch.softmax(m , -1)

def attention(Q, K, V):
    #Attention(Q, K, V) = norm(QK)V
    a = a_norm(Q, K) #(batch_size, dim_attn, seq_length)
    return  torch.matmul(a,  V) #(batch_size, seq_length, seq_length)

class AttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)
    
    def forward(self, x, kv = None):
        if(kv is None):
            #Attention with x connected to Q,K and V (For encoder)
            return attention(self.query(x), self.key(x), self.value(x))
        
        #Attention with x as Q, external vector kv as K an V (For decoder)
        return attention(self.query(x), self.key(kv), self.value(kv))
    
class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn))
        
        self.heads = nn.ModuleList(self.heads)
        
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias = False)
                      
    def forward(self, x, kv = None):
        a = []
        for h in self.heads:
            a.append(h(x, kv = kv))
            
        a = torch.stack(a, dim = -1) #combine heads
        a = a.flatten(start_dim = 2) #flatten all head outputs
        
        x = self.fc(a)
        
        return x
    
class Value(torch.nn.Module):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_val, bias = False)
    
    def forward(self, x):
        return self.fc1(x)

class Key(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = False)
    
    def forward(self, x):
        return self.fc1(x)

class Query(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = False)
    
    def forward(self, x):
        return self.fc1(x)

def QuantileLoss(net_out, Y, q):
    return (q * F.relu(net_out - Y)) + ((1 - q) * F.relu(Y - net_out))

from data import one_hot
def forward_pass(model, data_gen, batch_size, quantiles, indexer = None, gpu = True, one_hot_lens = [24, 31, 12], loss = True):
    if(type(data_gen) != type([])):
        data_gen = [data_gen]
    n_coins = len(data_gen)
    
    if(gpu):
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    #model.reset(batch_size, gpu = gpu)
    
    static = torch.arange(0, n_coins).unsqueeze(0).repeat(batch_size,1).transpose(1,0).flatten()
    static = static.unsqueeze(-1).unsqueeze(-1).float().type(dtype)
    static = one_hot(static, [5])
    
    in_seq_continuous, in_seq_discrete, future_in_seq_discrete, target_seq, mask  = next(data_gen[0])
    
    for gen in data_gen[1:]:
        n0, n1, n2, n3, n4  = next(gen)
        
        in_seq_continuous = np.concatenate((in_seq_continuous, n0), 0)
        in_seq_discrete = np.concatenate((in_seq_discrete, n1), 0)
        future_in_seq_discrete = np.concatenate((future_in_seq_discrete, n2), 0)
        target_seq = np.concatenate((target_seq, n3), 0)
        mask = np.concatenate((mask, n4), 0)
                                           
    
    
    in_seq_continuous = torch.tensor(in_seq_continuous).type(dtype).unsqueeze(-1)
    in_seq_discrete =  one_hot(torch.tensor(in_seq_discrete).type(dtype), one_hot_lens)
    future_in_seq_discrete = one_hot(torch.tensor(future_in_seq_discrete).type(dtype), one_hot_lens)
    target_seq = torch.tensor(target_seq).type(dtype)
    mask = torch.tensor(mask).type(dtype).unsqueeze(-1)
    
    #forward pass
    net_out, vs_weights, _ = model(in_seq_continuous, in_seq_discrete, None,
                                future_in_seq_discrete, static, mask, n_coins = n_coins)
    
    if loss:
        loss = torch.mean(QuantileLoss(net_out, target_seq ,quantiles), dim = -1)
        loss = loss.reshape([batch_size, n_coins, loss.shape[-1]])
        loss = torch.mean(loss, dim = -1)
        coin_losses = torch.mean(loss, dim = 0)
        loss = torch.mean(loss, dim = -1)
        if not (indexer is None):
            indexer.next(loss.cpu().detach().numpy())
        #loss = torch.mean(loss)



        #start_time = time.time()    
        #print("--- %s seconds ---" % (time.time() - start_time))
        return coin_losses, net_out, vs_weights, (in_seq_continuous, in_seq_discrete, future_in_seq_discrete, target_seq)
    else:
        if not (indexer is None):
            indexer.next()
        return net_out, vs_weights, (in_seq_continuous, in_seq_discrete, future_in_seq_discrete, target_seq)
    


