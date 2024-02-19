import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
import os
os.environ["OMP_NUM_THREADS"] = "8"

import sys

from pcan import PCAN,PCAN_CONFIG
from metrics import Weighted_Mean_Absolute_Percentage_Error,ACC,Mean_Absolute_Percentage_Error,Weighted_Square_Error
from Dataset import Dataset_final_fixed
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.nn.parallel import DistributedDataParallel,DataParallel

import torch
from loguru import logger
import argparse

import os
local_rank = int(os.environ["LOCAL_RANK"])


@logger.catch()
def train_val(epochs_1,epoch_2,learning_rate,batch_size = 128,nontion_item_cd = "NON10",things = "",mode = "all"):
    
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        'gloo',
        init_method='env://'
    )
    device = torch.device(f'cuda:{local_rank}')
    # initialize steps
    train_steps = 0
    val_steps = 0

    # initialize model
    
    model = torch.load("/home/huiyu/best_models/new/NON319/new_ep_1652700234.6970408/best_model.pth.tar" )

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    # hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    # hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # criterion & scheduler
    # criterion = Weighted_Square_Error(0.9).to("cuda")
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'max',factor = 0.1,patience = 3,cooldown = 5,min_lr  = 1e-8)

    # time stamp
    time_stamp = str(time.time())
    
    # initialize tensorboard
    writer = SummaryWriter(f"/home/huiyu/best_models/new/{nontion_item_cd}/{things}_{time_stamp}/")


    # record hyperparams
    writer.add_text("traing/learning_rate",str(learning_rate),0)
    writer.add_text("traing/epochs_1",str(epochs_1),0)
    writer.add_text("traing/batch_size",str(batch_size),0)
    writer.add_text("traing/nontion_item_cd",nontion_item_cd,0)
    writer.add_text("model/d_model",str(PCAN_CONFIG["sales_extractor_config"]["feature_size"]),0)
    writer.add_text("model/time_lenth",str(PCAN_CONFIG["sales_extractor_config"]["time_lenth"]),0)
    
    logger.info("start loading ds")
    # initialize dataset
    dstrain = Dataset_final_fixed(data_path = '/root/autodl-nas/perstore/',
                                config_path='/root/features_new.yaml',
                                train_horizon=30,test_horizon=7,slide_mode='per_day',
                                notional_item_cd=nontion_item_cd,period='train')
    

    train_sampler = torch.utils.data.distributed.DistributedSampler(dstrain)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(dstrain, num_replicas=hvd.size(), rank=hvd.rank())

    dsval = Dataset_final_fixed(data_path = '/root/autodl-nas/perstore/',
                                config_path='/root/features_new.yaml',
                                train_horizon=30,test_horizon=7,slide_mode='per_day',
                                notional_item_cd=nontion_item_cd,period='val')
    logger.info("ds loaded")
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(dsval)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(dsval, num_replicas=hvd.size(), rank=hvd.rank())

    # used for best model
    best_acc = 0

    # start traing & evaling
    for i in range(epochs_1+epoch_2):
        
        # make record folder
        path = f"/home/huiyu/best_models/new/{nontion_item_cd}/{things}_{time_stamp}/"
        # path = f"./matisu_results/{nontion_item_cd}/{things}_{time_stamp}/"
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        # start traing
        # dl = DataLoader(dstrain,batch_size=batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=10,drop_last = False)
        dl = DataLoader(dstrain,batch_size=batch_size, shuffle=False, sampler=train_sampler, batch_sampler=None, num_workers=10,drop_last = False)

        for step,data in enumerate(dl):
            
            train_steps += 1
            X,label = data

            sales_features_masked = X[:,:,:6]  # 6
            epidemic_features_masked = X[:,:,6:14]  # 8
            predictable_features = X[:,:,14:53]  # 39
            unpredictable_features_masked = X[:,:,53:57]  # 4
            static_features = X[:,-1,57:63]   # 6
            static_features = torch.squeeze(static_features,dim = 1)
            data = (sales_features_masked,epidemic_features_masked,predictable_features,unpredictable_features_masked,static_features)
            
            if epoch_2 == 0:
                out = model(data,mode)
            else:
                if i< epochs_1:
                    out = model(data,"atte")
                else:
                    out = model(data,"gaus")


            # label = label.view(label.shape[0],-1).to("cuda")
            label = label.to("cuda")
            loss = criterion(out, label)
            

            # cal mse & mae & wmape & mape & loss & acc
            with torch.no_grad():
                mse_loss = F.mse_loss(out,label)
                mae_loss = F.l1_loss(out,label)
                mape = Mean_Absolute_Percentage_Error(out,label)
                wmape = Weighted_Mean_Absolute_Percentage_Error(out,label)
                acc = ACC(out,label)
            

            # show & record  
            writer.add_scalar("train/loss",loss,train_steps)
            writer.add_scalar("train/mse",mse_loss,train_steps)
            writer.add_scalar("train/mae",mae_loss,train_steps)
            writer.add_scalar("train/mape",mape,train_steps)
            writer.add_scalar("train/wmape",wmape,train_steps)
            writer.add_scalar("train/acc",acc,train_steps)
            

            # wandb.log({"train_loss": loss/batch_size})     
            logger.info(f"train - epoch: {i} step:{step} loss:{loss}")

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        # record grad
        for name, param in model.named_parameters():
            try:
                writer.add_histogram(name + '_data', param, i)
            except:
                pass

        # start validating

        # dl = DataLoader(dsval,batch_size=64, shuffle=False, sampler=None, batch_sampler=None, num_workers=10,drop_last = False)
        dl = DataLoader(dsval,batch_size=64, shuffle=False, sampler=val_sampler, batch_sampler=None, num_workers=10,drop_last = False)
        
        all_mse_loss = 0
        all_mae_loss = 0
        loss_record = 0



        for step,data in enumerate(dl):
            
            val_steps += 1
            X,label = data

            sales_features_masked = X[:,:,:6]  # 6
            epidemic_features_masked = X[:,:,6:14]  # 8
            predictable_features = X[:,:,14:53]  # 39
            unpredictable_features_masked = X[:,:,53:57]  # 4
            static_features = X[:,-1,57:63]   # 6
            static_features = torch.squeeze(static_features,dim = 1)
            data = (sales_features_masked,epidemic_features_masked,predictable_features,unpredictable_features_masked,static_features)

            with torch.no_grad():
                model.eval()
                out = model(data,mode)

                #label = label.view(label.shape[0],-1).to("cuda")
                label = label.to("cuda")

                loss = criterion(out, label)
                
                mse_loss = F.mse_loss(out, label)
                mae_loss = F.l1_loss(out,label)
                mape = Mean_Absolute_Percentage_Error(out,label)
                wmape = Weighted_Mean_Absolute_Percentage_Error(out,label)
                acc = ACC(out,label)
                

                try:
                    out_record = torch.concat([out_record,out.detach()],dim = -1)
                except:
                    out_record = out.detach()
                    
                try:
                    label_record = torch.concat([label_record,label.detach()],dim = -1)
                except:
                    label_record = label.detach()
                    
                writer.add_scalar("val/mse",mse_loss,val_steps)
                writer.add_scalar("val/mae",mae_loss,val_steps)
                writer.add_scalar("val/loss",loss,val_steps)
                writer.add_scalar("val/mape",mape,val_steps)
                writer.add_scalar("val/wmape",wmape,val_steps)
                writer.add_scalar("val/acc",acc,val_steps)


                all_mse_loss += mse_loss
                all_mae_loss += mae_loss
                loss_record += loss


        # show & record 
        mse = F.mse_loss(out_record,label_record)
        mae = F.l1_loss(out_record,label_record)

        wmape = Weighted_Mean_Absolute_Percentage_Error(out_record,label_record)
        acc = ACC(out_record,label_record)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.module.state_dict(), path+f"best_model_module.pth.tar")
            torch.save(model.module, path+f"best_model.pth.tar")

        mape = Mean_Absolute_Percentage_Error(out_record,label_record)


        writer.add_scalar("val/mse_all",mse,i)
        writer.add_scalar("val/mae_all",mae,i)
        writer.add_scalar("val/mean_loss",loss_record/step,i)
        writer.add_scalar("val/mape_all",mape,i)
        writer.add_scalar("val/wmape_all",wmape,i)
        writer.add_scalar("val/acc_all",acc,i)
        
        # update scheduler
        writer.add_scalar("train/learning_rate",float(optimizer.param_groups[0]['lr']),i)
        scheduler.step(acc)
        # writer.add_scalar("train/learning_rate",str(optimizer.state_dict()['param_groups'][0]['lr']),i)

        # print logger
        logger.info(f"val - epoch: {i} step:{i} loss:{all_mse_loss/step}")

        del out_record,label_record

        # wandb.log({"val_loss": all_loss/step})

# tic_list = ['NON10','NON691', 'NON327722','NON615','NON683', 'NON319', 'NON309743', 'NON7600', 'NON3936']
tic_list = [ 'NON319']

for k in tic_list:
    logger.info(f"{k} is proccesing")
    try:
        train_val(100,0,1e-5,batch_size = 200,nontion_item_cd = k, things = "new_ep",mode = "atte")
    except:
        pass