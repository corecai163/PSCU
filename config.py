# -*- coding: utf-8 -*-
# @Author: XP

from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()

__C.DATASETS.SHAPENET                            = edict()
__C.DATASETS.SHAPENET.CATEGORY_FILE_PATH         = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.N_RENDERINGS               = 8
__C.DATASETS.SHAPENET.N_POINTS                   = 2048
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = '../data/PCN/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = '../data/PCN/%s/complete/%s/%s.pcd'

__C.DATASETS.PU1K                        = edict()
__C.DATASETS.PU1K.PARTIAL_POINTS_PATH    = '../data/PU1K/%s'
__C.DATASETS.PU1K.PARTIAL_PATH    = '../data/PU1K/%s/input/'
__C.DATASETS.PU1K.COMPLETE_PATH   = '../data/PU1K/%s/gt/'

__C.DATASETS.KITTIPCD                        = edict()
__C.DATASETS.KITTIPCD.PARTIAL_POINTS_PATH        = '../data/kitti/%s/'
__C.DATASETS.KITTIPCD.PARTIAL_PATH               = '../data/kitti/%s/'
__C.DATASETS.KITTIPCD.COMPLETE_POINTS_PATH       = '../data/kitti/%s/'
#
# Dataset
#
__C.DATASET                                      = edict()
# Dataset Options: Completion3D, ShapeNet, ShapeNetCars, KITTIPCD,PU1K
__C.DATASET.TRAIN_DATASET                        = 'PU1K'
__C.DATASET.TEST_DATASET                         = 'PU1K'

#
# Constants
#
__C.CONST                                        = edict()

__C.CONST.NUM_WORKERS                            = 16
__C.CONST.N_INPUT_POINTS                         = 2048
__C.CONST.N_OUTPUT_POINTS                         = 8192

#
# Directories
#
__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = './out_path'
__C.CONST.DEVICE                                 = '0,1,2,3'
__C.CONST.WEIGHTS                                = './out_path/checkpoints/ckpt-best.pth' # 'ckpt-best.pth'  # specify a path to run test and inference

#
# Memcached
#
__C.MEMCACHED                                    = edict()
__C.MEMCACHED.ENABLED                            = False
__C.MEMCACHED.LIBRARY_PATH                       = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG                      = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                      = '/mnt/lustre/share/memcached_client/client.conf'

#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.N_SAMPLING_POINTS                    = 2048

#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 16
__C.TRAIN.N_EPOCHS                               = 100
__C.TRAIN.SAVE_FREQ                              = 25
__C.TRAIN.LEARNING_RATE                          = 1e-3
__C.TRAIN.LR_MILESTONES                          = [50, 100, 150, 200, 250]
__C.TRAIN.LR_DECAY_STEP                          = 50
__C.TRAIN.WARMUP_STEPS                           = 200
__C.TRAIN.GAMMA                                  = .5
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0

#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'
