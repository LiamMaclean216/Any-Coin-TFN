import pandas as pd
import datetime
import torch
import torch.nn as nn
from statsmodels.tsa.stattools import adfuller
import numpy as np
import time

import random

import plotly.graph_objects as go

header = {'open time' : 1, 'open' : 2, 'high' : 3, 'low' : 4, 'close' : 5}

#load_data from file
def load_data(year, symbol, interval = "1m", normalise = True):
    if(type(year) != type([])):
        year = [year]
    
    if(type(symbol) != type([])):
        symbol = [symbol]
    
    data = []
    for s in symbol:
        frames = []
        for y in year:
            csv_data = pd.read_csv("data_{}/{}_{}.csv".format(interval, y, s))
            frames.append(csv_data)    
            
        data_ = pd.concat(frames)
        
        #convert timestamp into month and day numbers
        data_['Hour'] = pd.to_datetime(data_["Open Time"], unit='ms').apply(lambda x: x.hour)    
        data_['Day'] = pd.to_datetime(data_["Open Time"], unit='ms').apply(lambda x: x.day - 1)
        data_['Month'] = pd.to_datetime(data_["Open Time"], unit='ms').apply(lambda x: x.month - 1)
        data.append(data_)
    
    print("done")
    return data

def softmax(x, dim = 0):
    return np.exp(x) / np.sum(np.exp(x), axis=dim)

class Indexer():
    def __init__(self, r_bottom, r_top, batch_size, random = True, increment = 1):
        self.r_bottom = r_bottom
        self.r_top = r_top
        self.random = random
        self.increment = increment
        self.batch_size = batch_size
        self.indices = [0]
        self.losses = np.ones([r_top - r_bottom]) * 100
        self.next()
        
    def next(self, losses = None):
        if(self.random):
#             if (losses is not None):
#                 self.losses[np.array(self.indices) - self.r_bottom] = losses
#                 #new_indicies = self.losses.argsort()[-self.batch_size:][::-1] + self.r_bottom
#                 #self.indices = new_indicies

#                 new_indicies = np.arange(self.r_bottom, self.r_top)
#                 self.indices = np.random.choice(new_indicies , self.batch_size, p = softmax(self.losses ** (1/4), -1))
#                 return self.indices
                
            ####    
            new_indices = []
            for b in range(self.batch_size):
                new_indices.append(random.randrange(self.r_bottom, self.r_top))
            self.indices = new_indices
        else:
#             new_indices = [self.indices[-1]]
            
#             for b in range(1, self.batch_size):
#                 i = new_indices[-1] + self.increment
#                 if(i >= self.r_top):
#                     new_indices.append((i - self.top) + self.r_bottom)
#                 else:
#                     new_indices.append(i)
#             self.indices = new_indices
            self.indices = np.arange(self.indices[-1] + self.increment, self.indices[-1] + self.increment + self.batch_size, self.increment)
            
        return self.indices
    
def get_batches(data_, in_seq_len, out_seq_len, con_cols, disc_cols, target_cols, one_hot_lens = [24, 31, 12], batch_size = 1, gpu = True, normalise = True, indexer = None, norm = None):
    data = data_.copy()
    
    given_indexer = True
    if indexer is None:
        given_indexer = False
        indexer = Indexer(1, data.shape[0] - (in_seq_len + out_seq_len + 1), batch_size)
        
    if normalise:
        if norm is None:
            norm = data
        data[con_cols] = (data[con_cols] - norm[con_cols].stack().mean()) / norm[con_cols].stack().std()
    
    #convert columns indices from dataframe to numpy darray
    con_cols = [data.columns.get_loc(x) for x in con_cols]
    disc_cols = [data.columns.get_loc(x) for x in disc_cols]
    target_cols = [data.columns.get_loc(x) for x in target_cols]
    
    if(not gpu):
        dtype = torch.FloatTensor
    else:
        dtype = torch.cuda.FloatTensor
        
    while True:
        #get batches
        n = np.array([pd.np.r_[i:(i + in_seq_len + out_seq_len)] for i in indexer.indices])
        batch_data = data.iloc[n.flatten()].values
        batch_data = torch.tensor(batch_data.reshape(batch_size ,in_seq_len + out_seq_len, data.shape[-1]))
        
        mask = ~torch.isnan(batch_data).any(2)
        batch_data[~mask] = 0
        
        #split up batch data
        in_seq_continuous = batch_data[:, 0:in_seq_len, con_cols]
        in_seq_discrete = batch_data[:, 0:in_seq_len, disc_cols]

        out_seq = batch_data[:,in_seq_len:in_seq_len + out_seq_len, disc_cols]
        target_seq = batch_data[:,in_seq_len:in_seq_len + out_seq_len, target_cols]
        
        yield in_seq_continuous, in_seq_discrete, out_seq, target_seq, mask
        
        if(not given_indexer):
            indexer.next()
    
def one_hot(x, dims, gpu = True):
    out = []
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    
    if(not gpu):
        dtype = torch.FloatTensor
    else:
        dtype = torch.cuda.FloatTensor
        
    for i in range(0, x.shape[-1]):
        x_ = x[:,:,i].byte().cpu().long().unsqueeze(-1)
        o = torch.zeros([batch_size, seq_len, dims[i]]).long()

        o.scatter_(-1, x_,  1)
        out.append(o.type(dtype))
    return out