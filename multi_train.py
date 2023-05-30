# -*- coding: utf-8 -*-
# @Author: Pingping Cai
import shutil
import logging
import os
import torch
import numpy as np
import argparse
import utils.data_loaders
import utils.helpers
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from test import test_net

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.optim.lr_scheduler import StepLR
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import get_loss
from models.model import Upsample_Net as Model
from config import cfg
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def train_net(rank,num_gpus):
    # set_seed(1+args.local_rank)
    set_seed(5+rank)
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    ## load data
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_dataset=train_dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TRAIN)
    batch_size = cfg.TRAIN.BATCH_SIZE
    train_sampler = DistributedSampler(train_dataset, num_replicas=num_gpus, rank=rank)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    sampler = train_sampler,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=utils.data_loaders.collate_fn,
                                                    pin_memory=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
                                                  batch_size=batch_size//2,
                                                  num_workers=cfg.CONST.NUM_WORKERS//2,
                                                  collate_fn=utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  shuffle=False)



    model = Model(dim_feat=512)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        #setup(rank,num_gpus)
        #dist.init_process_group("nccl", rank=rank, world_size=num_gpus)
        model = model.to(rank)
        model = DDP(model,device_ids=[rank],output_device=rank, find_unused_parameters=True)

    if dist.get_rank() == 0:
        # Set up folders for logs and checkpoints
        output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
        cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
        cfg.DIR.LOGS = output_dir % 'logs'
        if not os.path.exists(cfg.DIR.CHECKPOINTS):
            os.makedirs(cfg.DIR.CHECKPOINTS)
        # backup model
        savefile = cfg.DIR.CHECKPOINTS+'/model.py'
        shutil.copyfile('models/model.py', savefile)
        shutil.copyfile('config.py', cfg.DIR.CHECKPOINTS+'/config.py')
        # Create tensorboard writers
        tensor_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    # Create the optimizers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=cfg.TRAIN.LEARNING_RATE,
                                       weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                       betas=cfg.TRAIN.BETAS)

    # lr scheduler
    scheduler_steplr = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                          after_scheduler=scheduler_steplr)

    init_epoch = 0
    best_metrics = float('inf')
    steps = 0

    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS,map_location=torch.device("cpu"))
        #checkpoint = torch.load(cfg.CONST.WEIGHTS)
        best_metrics = checkpoint['best_metrics']
        model.load_state_dict(checkpoint['model'])
        logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' % (init_epoch, best_metrics))
        #if args.resume_path: 
		#checkpoint = torch.load(args.resume_path, map_location=torch.device("cpu"))  
		#model.load_state_dict(checkpoint["state_dict"])
        #optimizer.load_state_dict(checkpoint["optimizer"])

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
#        batch_time = AverageMeter()
#        data_time = AverageMeter()
        #train_sampler.set_epoch(epoch_idx)
        model.train()
        
        total_cd_pc = 0
        total_cd_p1 = 0
        total_cd_p2 = 0
        #total_cd_p3 = 0
        total_dz = 0
        #batch_end_time = time()
        n_batches = len(train_data_loader)
        with tqdm(train_data_loader) as t:
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):
                
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                optimizer.zero_grad()
                partial = data['partial_cloud']
                gt = data['gtcloud']
                #print(partial.size())
                #print(gt.size())
                
                pcds_pred = model(partial)
                
                loss_total, losses,rot_m = get_loss(pcds_pred, partial, gt, sqrt=False)
                
                loss_total.backward()
                
                optimizer.step()
                
                cd_pc_item = losses[0].item() * 1e3
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e3
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e3
                total_cd_p2 += cd_p2_item
                #cd_p3_item = losses[3].item() * 1e3
                #total_cd_p3 += cd_p3_item
                dz_item = losses[3].item()
                total_dz += dz_item

                t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_p1_item, cd_p2_item, dz_item]])
                #torch.nn.utils.clip_grad_norm_(model.parameters(),10)
                if steps <= cfg.TRAIN.WARMUP_STEPS:
                    lr_scheduler.step()
                    steps += 1

        avg_cdc = total_cd_pc / n_batches
        avg_cd1 = total_cd_p1 / n_batches
        avg_cd2 = total_cd_p2 / n_batches
        #avg_cd3 = total_cd_p3 / n_batches
        avg_dz = total_dz / n_batches

        lr_scheduler.step()
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        #epoch_end_time = time()
        if dist.get_rank() ==0:
            tensor_writer.add_scalar('Train/Epoch/cd_pc', avg_cdc, epoch_idx)
            tensor_writer.add_scalar('Train/Epoch/cd_p1', avg_cd1, epoch_idx)
            tensor_writer.add_scalar('Train/Epoch/cd_p2', avg_cd2, epoch_idx)
            #tensor_writer.add_scalar('Train/Epoch/cd_p3', avg_cd3, epoch_idx)
            tensor_writer.add_scalar('Train/Epoch/dz', avg_dz, epoch_idx)
            logging.info(
            '[Epoch %d/%d] Losses = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, ['%.4f' % l for l in [avg_cd1, avg_cd2, avg_dz]]))

            # Validate the current model
            cd_eval = test_net(cfg, epoch_idx, val_data_loader, tensor_writer, model)

            # Save checkpoints
            if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
                file_name = 'ckpt-best.pth' if cd_eval < best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
                output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
                torch.save({
                    'epoch_index': epoch_idx,
                    'best_metrics': best_metrics,
                    'model': model.state_dict()
                }, output_path)
    
                logging.info('Saved checkpoint to %s ...' % output_path)
                if cd_eval < best_metrics:
                    best_metrics = cd_eval
    if dist.get_rank() ==0:
        tensor_writer.close()
    
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
if __name__ == '__main__':
    n_gpus = torch.cuda.device_count()
    dist.init_process_group(backend='nccl')
#    mp.spawn(train_net,
#             args=(n_gpus,),
#             nprocs=n_gpus,
#             join=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
    args = parser.parse_args()
    set_seed(5)
    train_net(args.local_rank,n_gpus)

