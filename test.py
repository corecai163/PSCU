# -*- coding: utf-8 -*-
# @Author: XP

import logging
import torch
import os
import numpy as np
import utils.data_loaders
import utils.helpers
from tqdm import tqdm
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from utils.loss_utils import get_loss
from models.model import Upsample_Net as Model
import open3d as o3d
from config import cfg

def save_pcd(path,name,xyz):
    ## save pcd
    save_path = os.path.join(path,name)
#    print(save_path)
#    print(xyz)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
    o3d.io.write_point_cloud(save_path,pcd)
   
    
def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetSubset.TEST),
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.NUM_WORKERS,
                                                       collate_fn=utils.data_loaders.collate_fn,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Setup networks and initialize networks
    if model is None:
        model = Model(dim_feat=512)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])

    # Switch models to evaluation mode
    model.eval()

    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['cdc','cd1', 'cd2', 'dz'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    # INIT TIME LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((n_samples,1))

    # Testing loop
    with tqdm(test_data_loader) as t:
        for model_idx, (taxonomy_id, name, data) in enumerate(t):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            model_id = name[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                partial = data['partial_cloud']
                gt = data['gtcloud']

                b, n, _ = partial.shape

                starter.record()
                pcds_pred = model(partial.contiguous())
                ender.record()

                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[model_idx] = curr_time


                loss_total, losses,rot_m = get_loss(pcds_pred, partial, gt, sqrt=False)

                ## save pred pcds
                #save_pcd('results',name[0],pcds_pred[2].squeeze().cpu().numpy())
                cdc = losses[0].item() * 1e3
                cd1 = losses[1].item() * 1e3
                cd2 = losses[2].item() * 1e3
                dz = losses[3].item() * 1e3
                #rot = losses[3].item() * 1e3

                _metrics = [cd2]
                test_losses.update([cdc,cd1, cd2, dz])

                test_metrics.update(_metrics)
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

                t.set_description('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                             (model_idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()
                                                                                ], ['%.4f' % m for m in _metrics]))

    # Print testing results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

#    for taxonomy_id in category_metrics:
#        print(taxonomy_id, end='\t')
#        print(category_metrics[taxonomy_id].count(0), end='\t')
#        for value in category_metrics[taxonomy_id].avg():
#            print('%.4f' % value, end='\t')
#        print()

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    print('Epoch ', epoch_idx, end='\t')
    for value in test_losses.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    print('Average Running Time ', np.mean(timings))

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Test/Epoch/cdc', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Test/Epoch/cd1', test_losses.avg(1), epoch_idx)
        test_writer.add_scalar('Test/Epoch/cd2', test_losses.avg(2), epoch_idx)
        test_writer.add_scalar('Test/Epoch/dz', test_losses.avg(3), epoch_idx)
        #test_writer.add_scalar('Test/Epoch/rot', test_losses.avg(3), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return test_losses.avg(2)

if __name__ == '__main__':
    test_net(cfg)
