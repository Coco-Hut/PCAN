import torch
from torch import nn
import math

# data embedding
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

# positional embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# timestamp embedding
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model,device = "cuda"):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model,device = device).float()
        w.require_grad = False

        position = torch.arange(0, c_in,device = device).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2,device = device).float() * -(math.log(10000.0) / d_model)).exp()


        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixe', freq='h',device = "cuda"):
        super(TemporalEmbedding, self).__init__()

        year_size = 2024; half_size = 3; quarter_size = 5; month_size = 13; mday_size = 32
        qday_size = 100; yday_size = 600; week_size  = 200; mweek_size = 6; wday_size = 8

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        self.year_embed = Embed(year_size, d_model,device = device)
        self.half_embed = Embed(half_size, d_model,device = device)
        self.quarter_embed = Embed(quarter_size, d_model,device = device)
        self.month_embed = Embed(month_size, d_model,device = device)
        self.mday_embed = Embed(mday_size, d_model,device = device)
        self.qday_embed = Embed(qday_size, d_model,device = device)
        self.yday_embed = Embed(yday_size, d_model,device = device)
        self.week_embed = Embed(week_size, d_model,device = device)
        self.mweek_embed = Embed(mweek_size, d_model,device = device)
        self.wday_embed = Embed(wday_size, d_model,device = device)

    def forward(self, x):
        x = x.long()

        year_x = self.year_embed(x[:,:,0])  # if hasattr(self, 'minute_embed') else 0.
        half_x = self.half_embed(x[:,:,1])
        quarter_x = self.quarter_embed(x[:,:,2])
        month_x = self.month_embed(x[:,:,3])
        mday_x = self.mday_embed(x[:,:,4])
        qday_x = self.qday_embed(x[:,:,5])
        yday_x = self.yday_embed(x[:,:,6])
        week_x = self.week_embed(x[:,:,7])
        mweek_x = self.mweek_embed(x[:,:,8])
        wday_x = self.wday_embed(x[:,:,9])
        
        return year_x + half_x + quarter_x + month_x + mday_x +\
            qday_x + yday_x + week_x + mweek_x + wday_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

# Data Embeding old
class DataEmbedding_old(nn.Module):
    def __init__(self, config,type,dropout=0.1):
        super(DataEmbedding_old, self).__init__()
        self.feature_embedding = TokenEmbedding(config["Other_features_in"],config["Other_features_out"])
        self.epidemic_embedding = TokenEmbedding(config["Epidemic_features"],1)
        self.positional_embedding = PositionalEmbedding(512)
        self.type = type
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,sales,history_features,future_features,epidemic_features = None,device = "cpu"):
        if self.type == "decoder":
            days = future_features.shape[1]-history_features.shape[1]
            zeros = torch.zeros(history_features.shape[0],days,history_features.shape[2],device = device)
            history_features = torch.cat([history_features,zeros],1)
        other_features_encoder = torch.cat([history_features,future_features],2)
        other_features_encoder = self.feature_embedding(other_features_encoder)
        other_features_encoder = torch.cat([sales,other_features_encoder],2)
        positional_embeddings = self.positional_embedding(other_features_encoder)
        other_features_encoder += positional_embeddings
        if self.type == "encoder":
            epidemic_features = self.epidemic_embedding(epidemic_features)
            epidemic_features = torch.repeat_interleave(epidemic_features, repeats=other_features_encoder.shape[2] , dim=2)
            other_features_encoder += epidemic_features
        return self.dropout(other_features_encoder)

# fake embedding
class FakeEmbedding(nn.Module):
    def __init__(self, d_in, d_out):
        super(FakeEmbedding, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.emb = nn.Linear(d_in,d_out)

    def forward(self,x):
        return self.emb(x)

# static embedding
class StaticEmbedding(nn.Module):
    def __init__(self,d_in,d_out,time_lenth):
        super(StaticEmbedding, self).__init__()
        self.d_in = d_in
        self.time_lenth = time_lenth
        self.d_out = d_out
        self.emb_list = nn.ModuleList()
        for i in range(d_in):
            self.emb_list.append(nn.Embedding(1000,d_out))
        

    def forward(self,x, time_lenth=40,device = "cuda"):
        # only support input size (batchsize, features)
        x = x.long()
        if self.d_in != x.shape[-1]:
            raise ValueError("Please check your input size!")
        
        if len(x.shape) == 2:
            out = torch.zeros(x.shape[0],self.d_out,device=device)
        else:
            raise NotImplementedError("only support input size (batchsize, features)")
            
            
        for i in range(len(self.emb_list)):
            temp_tensor = x[:,i]
            temp_tensor = temp_tensor.long()
            out += self.emb_list[i](temp_tensor)

        out = out.view(out.shape[0],-1,out.shape[-1])
        out = out.repeat(1,40,1)
        # out = torch.repeat_interleave(out,self.time_lenth,dim = 1)
        
        return out

# data embedding
class DataEmbedding(nn.Module):
    def __init__(self,embedding_config):
        super(DataEmbedding, self).__init__()
        self.sales_embedding = TokenEmbedding(embedding_config["sales_features"],
                                            embedding_config["atte_out_dim"])
        self.epidemic_embedding = TokenEmbedding(embedding_config["epidemic_features"],
                                            embedding_config["atte_out_dim"])
        self.covariate_embedding = TokenEmbedding(embedding_config["predictable_features"]+embedding_config["unpredictable_features"],
                                                embedding_config["atte_out_dim"])

        self.extractor_positional_embedding = PositionalEmbedding(embedding_config["atte_out_dim"])

        
        

    def forward(self,sales_features,epidemic_features,predictable_features,unpredictable_features):
        sales_out = self.sales_embedding(sales_features) + self.extractor_positional_embedding(sales_features)
        epidemic_out = self.epidemic_embedding(epidemic_features) + self.extractor_positional_embedding(epidemic_features)
        covariate_features = torch.concat([predictable_features,unpredictable_features],dim = -1)  
        covariate_out = self.covariate_embedding(covariate_features) + self.extractor_positional_embedding(covariate_features)
        
        return sales_out,epidemic_out,covariate_out

# sale embedding
class SalesEmbedding(nn.Module):
    def __init__(self,embedding_config,device = "cuda"):
        super(SalesEmbedding, self).__init__()
        self.sales_embedding = TokenEmbedding(embedding_config["sales_features"],
                                            embedding_config["decoder_dim"])

        self.temporal_embedding = TemporalEmbedding(embedding_config["decoder_dim"],device = device)

        self.static_embedding = StaticEmbedding(embedding_config["static_features"],
                                            embedding_config["static_out_features"],
                                            embedding_config["time_lenth"])

    def forward(self,sales,timefeatures,static,mask_len = 7):
        sales  = sales[:,:-mask_len,:]
        sales = self.sales_embedding(sales)
        zero_features = torch.zeros(sales.shape[0],mask_len,sales.shape[-1],device = sales.device)
        sales = torch.concat([sales,zero_features],dim = 1)
        X = sales + self.temporal_embedding(timefeatures) + self.static_embedding(static)
        return X


# matisu embedding
class MatisuEmbedding(nn.Module):
    def __init__(self,embedding_config,device = "cuda"):
        super(MatisuEmbedding, self).__init__()
        self.sales_mask_len = embedding_config["sales_mask_len"]
        self.sales_embedding = TokenEmbedding(embedding_config["sales_features"],
                                            embedding_config["out_dim"])
        self.epidemic_embedding = TokenEmbedding(embedding_config["epidemic_features"],
                                            embedding_config["out_dim"])
        self.covariate_embedding = TokenEmbedding(embedding_config["predictable_features"]+embedding_config["unpredictable_features"],
                                                embedding_config["out_dim"])
        self.temporal_embedding = TemporalEmbedding(embedding_config["out_dim"],device = device)
        self.static_embedding = StaticEmbedding(embedding_config["static_features"],
                                            embedding_config["out_dim"],
                                            embedding_config["time_lenth"])

        self.extractor_positional_embedding = PositionalEmbedding(embedding_config["out_dim"])

    def forward(self,sales,epidemic,predictable,unpredictable,timefeatures,static):
        # sales
        sales  = sales[:,:-self.sales_mask_len,:]
        sales = self.sales_embedding(sales)
        sales_zero = torch.zeros(sales.shape[0],self.sales_mask_len,sales.shape[-1],device = sales.device)
        sales = torch.concat([sales,sales_zero],dim = 1)
        sales += self.extractor_positional_embedding(sales)

        # epidmic
        epidemic = self.epidemic_embedding(epidemic) + \
                    self.extractor_positional_embedding(epidemic)
        
        # covariate (predictable,unpredictable)
        covariate = torch.concat([predictable,unpredictable],dim = -1)  
        covariate = self.covariate_embedding(covariate) + \
                        self.extractor_positional_embedding(covariate)

        # timefeatures
        timefeatures = self.temporal_embedding(timefeatures)

        # static
        static = self.static_embedding(static)
        
        return sales,epidemic,covariate,timefeatures,static