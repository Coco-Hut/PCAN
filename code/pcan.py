import torch
from torch import nn
import pandas as pd
import sys
from embedding import MatisuEmbedding
from pcan_blocks import BaseAttention,FluctuationAttention,EventDecoder,EnentEncoderDecoder,PostProcessing

class PCAN(nn.Module):
    def __init__(self, model_config,device = "cuda"):
        super(PCAN, self).__init__()

        # embedding
        self.embedding = MatisuEmbedding(model_config["embedding_config"],device = device)

        # encoder_decoder
        covariate_extractor = BaseAttention(model_config["covariate_extractor_config"])
        event_extractor = BaseAttention(model_config["event_extractor_config"])
        sales_extractor = FluctuationAttention(model_config["sales_extractor_config"])
        event_decoder = EventDecoder(model_config["event_decoder_config"])

        self.encoder_decoder = EnentEncoderDecoder(covariate_extractor,\
            event_extractor,sales_extractor,event_decoder,\
                model_config["covariate_extractor_config"])

        # postprocessing
        self.postprocessing = PostProcessing(model_config["postprocessing_config"])

        self.device = device

    def forward(self,data,mode = "all"):
        sales,epidemic,predictable,unpredictable,static = data

        sales = sales.to(self.device)
        epidemic = epidemic.to(self.device)
        predictable = predictable.to(self.device)
        unpredictable = unpredictable.to(self.device)
        static = static.to(self.device)

        timefeatures = predictable[:,:,:10]

        predictable = predictable[:,:,10:]

        # Embeddings
        sales,epidemic,covariate,timefeatures,static = self.embedding(sales,\
            epidemic,predictable,unpredictable,timefeatures,static)

        Cov_X = (sales,covariate)
        Event_x = (sales,epidemic)
        Sales_X = sales+timefeatures+static

        # computing
        out = self.encoder_decoder(Cov_X,Event_x, Sales_X,self.device)
        pred_result = self.postprocessing(out)

        return pred_result



PCAN_CONFIG = {
    'embedding_config': {
        'sales_features': 6,
        'epidemic_features': 8,
        'predictable_features': 29,
        'unpredictable_features': 4,
        'static_features': 6,

        'out_dim': 512,
        
        "time_lenth":40,

        "sales_mask_len":7
        # "time_features":10,
        # "decoder_dim":512,
        },

    'covariate_extractor_config': {
        
        'feature_size': 512,
        'num_hiddens': 512,      
        'ffn_num_hiddens': 512,  
        'num_heads': 8,
        'time_lenth': 40,        # total timesteps
        'use_bias': True,         
        'num_layers':6,          # number of blocks
        'dropout': 0.5,
        },

    'event_extractor_config': {
        
        'feature_size': 512,
        'num_hiddens': 512,      
        'ffn_num_hiddens': 512, 
        'num_heads': 8,
        'time_lenth': 37,        # total timesteps
        'use_bias': True,         
        'num_layers':6,          # number of blocks
        'dropout': 0.5,
        },
    
    'sales_extractor_config': {
        
        'feature_size': 512,
        'num_hiddens': 512,      
        'ffn_num_hiddens': 512,  
        'num_heads': 8,
        'time_lenth': 37,        # total timesteps
        'use_bias': True,         
        'num_layers':6,          # number of blocks
        'dropout': 0.5,
        },

    'event_decoder_config': {
        'feature_size': 512,
        'num_hiddens': 512,      
        'ffn_num_hiddens': 512,  
        'num_heads': 8,
        'time_lenth': 37,        # total timesteps
        'use_bias': True,         
        'num_layers':6,          # number of blocks
        'dropout': 0.5,
        "g_out":240
        },

    
    'postprocessing_config': {
        'input_days': 40,
        'input_features': 512,
        'output_days': 7,
        }
}