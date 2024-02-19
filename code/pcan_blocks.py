from os import X_OK
import torch
from torch import nn
import sys
from attentions import MultiHeadAttention,Anomaly_MultiHeadAttention,AddNorm,PositionWiseFFN

# Event & Covariate
class BaseAttention_Block(nn.Module):
    def __init__(self,attention_config):
        super(BaseAttention_Block, self).__init__()
        self.atte_1 = MultiHeadAttention(key_size = attention_config["feature_size"], 
                                            query_size = attention_config["feature_size"], 
                                            value_size = attention_config["feature_size"], 
                                            num_hiddens = attention_config["num_hiddens"],
                                            num_heads = attention_config["num_heads"], 
                                            dropout = attention_config["dropout"], 
                                            bias=False)
        self.add_norm_1 = AddNorm(normalized_shape = [attention_config["time_lenth"],
                                                attention_config["feature_size"]],
                                                dropout= attention_config["dropout"])

        self.atte_2 = MultiHeadAttention(key_size = attention_config["feature_size"], 
                                            query_size = attention_config["feature_size"], 
                                            value_size = attention_config["feature_size"], 
                                            num_hiddens = attention_config["num_hiddens"],
                                            num_heads = attention_config["num_heads"], 
                                            dropout = attention_config["dropout"], 
                                            bias=False)
        self.add_norm_2 = AddNorm(normalized_shape = [attention_config["time_lenth"],
                                                    attention_config["feature_size"]],
                                                    dropout= attention_config["dropout"])

        self.dense = PositionWiseFFN(ffn_num_input = attention_config["feature_size"], 
                                    ffn_num_hiddens = attention_config["ffn_num_hiddens"], 
                                    ffn_num_outputs = attention_config["feature_size"])

        self.add_norm_3 = AddNorm(normalized_shape = [attention_config["time_lenth"],
                                                     attention_config["feature_size"]],
                                                    dropout= attention_config["dropout"])

    def forward(self,Q,K,V,partical_mask = None):
        out = self.add_norm_1(K , self.atte_1(Q,K,V,partical_mask))
        out = self.add_norm_2(out , self.atte_2(out,out,out) )
        out = self.add_norm_2(out , self.dense(out) )
        return out

# Event & Covariate
class BaseAttention(nn.Module):
    def __init__(self,attention_config):
        super(BaseAttention, self).__init__()
        self.blocks = nn.Sequential()
        for i in range(attention_config["num_layers"]):
            self.blocks.add_module("block"+str(i),BaseAttention_Block(attention_config))

    def forward(self,data,mask = None):
        Q,X= data
        self._attention_weights = [[None] * len(self.blocks) for _ in range (2)]
        for i, blk in enumerate(self.blocks):
            X = blk(Q,X,X,mask)
            # att1 注意力权重
            self._attention_weights[0][i] = blk.atte_1.attention.attention_weights.cpu()
            # att2 自注意力权重
            self._attention_weights[1][i] = blk.atte_2.attention.attention_weights.cpu()

        return X

# Sales 
class FluctuationAttention_block(BaseAttention_Block):
    def __init__(self,attention_config):
        super(FluctuationAttention_block, self).__init__(attention_config)
        self.atte_1 = Anomaly_MultiHeadAttention(attention_config["feature_size"], 
                                                    attention_config["feature_size"], 
                                                    attention_config["feature_size"], 
                                                    attention_config["num_hiddens"], 
                                                    attention_config["num_heads"], 
                                                    attention_config["dropout"],)
        time_lenth = attention_config["time_lenth"]
        self.dense_g = nn.Sequential(
            nn.Linear(time_lenth,time_lenth),
            nn.SELU(),
            nn.Dropout(attention_config["dropout"]),
            nn.Linear(time_lenth,time_lenth),
        )

    
    def forward(self,Q,K,V,mask = 7,device = "cpu",mode = "all"):
        # anomaly self-attention
        X_out = self.add_norm_1(K, self.atte_1(Q, K, V,device = device,mode = mode))
        
        # multihead attention
        X_out = self.add_norm_2(X_out, self.atte_2(X_out, X_out, X_out))

        # FFN
        X_out = self.add_norm_3(X_out, self.dense(X_out))

        # fluctuation
        G_out = self.atte_1.association_discrepancy
        G_out = self.dense_g(G_out)

        # # matmul
        # X_5 = X_5.transpose(1,2)
        # Y = torch.unsqueeze(Y,dim = 1)

        # detach
        if mode == "atte":
            G_out = G_out.detach()
        
        if mode == "gaus":
            X_out = X_out.detach()
        
        # out  = (X_5*Y).transpose(1,2)

        self._G_out = G_out

        return X_out

    @property
    def G_out(self):
        return self._G_out

# Sales 
class FluctuationAttention(nn.Module):
    def __init__(self,attention_config):
        super(FluctuationAttention, self).__init__()
        self.blocks = nn.Sequential()
        for i in range(attention_config["num_layers"]):
            self.blocks.add_module("block"+str(i),FluctuationAttention_block(attention_config))
    
    def forward(self,X,mask = None,device = "cpu",mode = "all"):
        self._attention_weights = [[None] * len(self.blocks) for _ in range (2)]
        self._piror = [None] * len(self.blocks)
        self._G_out= [None] * len(self.blocks)
        for i, blk in enumerate(self.blocks):
            X = blk(X,X,X,mask,device,mode)
            self._attention_weights[0][i] = blk.atte_1.series.detach().cpu()
            self._piror[i] = blk.atte_1.prior.detach().cpu()
            self._G_out[i] = blk.G_out.detach()
            self._attention_weights[1][i] = blk.atte_2.attention.attention_weights.cpu()

        return X

    @property
    def attention_weights(self):
        return self._attention_weights

    @property
    def piror(self):
        return self._piror

    @property
    def G_out(self):
        return self._G_out

# decoder_old
class EventDecoder_block_old(BaseAttention_Block):
    def __init__(self,attention_config):
        super(EventDecoder_block_old, self).__init__(attention_config)
        
    def forward(self,X,state,G_out,partical_mask = None):
        enc_outputs, enc_valid_lens = state[0], state[1]

        # self-attention
        X_1 = self.atte_1(X, X, X)
        X_2 = self.add_norm_1(X, X_1)
        
        # multihead attention
        X_3 = self.atte_2(X_2, enc_outputs, enc_outputs, enc_valid_lens)
        X_4 = self.add_norm_2(X_2, X_3)

        # FFN
        out = self.add_norm_3(X_4, self.dense(X_4))

        # matmul
        out = out.transpose(1,2)
        G_out = torch.unsqueeze(G_out,dim = 1)
        out  = (out*G_out).transpose(1,2)

        return out


# decoder_old
class EventDecoder_old(nn.Module):
    def __init__(self,attention_config):
        super(EventDecoder_old, self).__init__()
        self.blocks = nn.Sequential()
        self.num_layers = attention_config["num_layers"]
        for i in range(attention_config["num_layers"]):
            self.blocks.add_module("block"+str(i),EventDecoder_block_old(attention_config))

    def forward(self, X ,state,G_out,device = "cpu"):
        self._attention_weights = [[None] * len(self.blocks) for _ in range (2)]

        for i, blk in enumerate(self.blocks):
            G = G_out[i]
            X = blk(X,state,G)
            self._attention_weights[0][i] = blk.atte_1.attention.attention_weights.cpu()
            self._attention_weights[1][i] = blk.atte_2.attention.attention_weights.cpu()
        
        return X
    
    def init_state(self, enc_outputs, enc_valid_lens = None, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    @property
    def attention_weights(self):
        return self._attention_weights

# decoder 
class EventDecoder_block(BaseAttention_Block):
    def __init__(self,attention_config):
        super(EventDecoder_block, self).__init__(attention_config)
        lenth = attention_config["g_out"]
        self.dense_g = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lenth,lenth),
            nn.SELU(),
            nn.Dropout(attention_config["dropout"]),
            nn.Linear(lenth,lenth),
            nn.SELU(),
            nn.Dropout(attention_config["dropout"]),
            nn.Linear(lenth,attention_config['time_lenth']),
        )
        
    def forward(self,X,state,G_out,partical_mask = None):
        enc_outputs, enc_valid_lens = state[0], state[1]

        # self-attention
        X_1 = self.atte_1(X, X, X)
        X_2 = self.add_norm_1(X, X_1)
        
        # multihead attention
        X_3 = self.atte_2(X_2, enc_outputs, enc_outputs, enc_valid_lens)
        X_4 = self.add_norm_2(X_2, X_3)

        # FFN
        out = self.add_norm_3(X_4, self.dense(X_4))

        # matmul
        out = out.transpose(1,2)
        G_out = self.dense_g(G_out)
        self.G_out = G_out
        G_out = torch.unsqueeze(G_out,dim = 1)
        out  = (out*G_out).transpose(1,2)

        return out
    

# decoder
class EventDecoder(nn.Module):
    def __init__(self,attention_config):
        super(EventDecoder, self).__init__()
        self.blocks = nn.Sequential()
        self.num_layers = attention_config["num_layers"]
        for i in range(attention_config["num_layers"]):
            self.blocks.add_module("block"+str(i),EventDecoder_block(attention_config))

    def forward(self, X ,state,G_out,device = "cpu"):
        self._attention_weights = [[None] * len(self.blocks) for _ in range (2)]

        for i, blk in enumerate(self.blocks):
            G = torch.stack(G_out).permute(1,0,2)
            X = blk(X,state,G)
            self._attention_weights[0][i] = blk.atte_1.attention.attention_weights.cpu()
            self._attention_weights[1][i] = blk.atte_2.attention.attention_weights.cpu()
        
        return X
    
    def init_state(self, enc_outputs, enc_valid_lens = None, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    @property
    def attention_weights(self):
        return self._attention_weights


# Encoder Decoder
class EnentEncoderDecoder(nn.Module):
    def __init__(self, CovariateExtractor,EventExtractor, SalesExtractor,EventDecoder, config,**kwargs):
        super(EnentEncoderDecoder, self).__init__(**kwargs)
        self.norm = nn.LayerNorm( [config["time_lenth"],config["feature_size"]])
        self.CovariateExtractor = CovariateExtractor
        self.EventExtractor = EventExtractor
        self.SalesExtractor = SalesExtractor
        self.EventDecoder = EventDecoder

    def forward(self, Cov_X, Event_x,Sales_X,device = "cpu",mode = "all",*args):
        Cov_out = self.CovariateExtractor(Cov_X)
        Sal_out = self.SalesExtractor(Sales_X,device = device)
        G_out = self.SalesExtractor.G_out
        Eve_out = self.EventExtractor(Event_x)
        dec_state = self.EventDecoder.init_state(Eve_out)

        dev_in = self.norm(Cov_out+Sal_out)

        return self.EventDecoder(dev_in,dec_state,G_out,device)

class PostProcessing(nn.Module):
    def __init__(self,config, **kwargs):
        super(PostProcessing, self).__init__(**kwargs)
        
        
        self.dense_1 = nn.Linear(config["input_days"],config["output_days"])
        self.act_fn = nn.SELU()
        self.dropout = nn.Dropout()
        self.dense_2 = nn.Linear(config["input_features"],1)

    def forward(self, decoder_output):
        X = decoder_output.transpose(1,2)
        X = self.act_fn(self.dense_1(X))
        X = self.dropout(X)
        X = X.transpose(1,2)
        X = self.dense_2(X)

        return X