import argparse
import numpy as np
import os
import random


def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument("--dataset", type=str, default='syn', help="Dataset string")# 'Mutagenicity'
    parser.add_argument('--id', type=str, default='default_id', help='id to store in database')  #
    parser.add_argument('--device', type=str, default='0',help='device to use')  #
    parser.add_argument('--early_stop', type=int, default= 100, help='early_stop')
    parser.add_argument('--dtype', type=str, default='float32')  #
    parser.add_argument('--seed',type=int, default=1234, help='seed')
    parser.add_argument('--setting',type=int, default=2, help='setting')

    parser.add_argument('--order', type=str, default='AW')  #
    parser.add_argument('--embnormlize', type=bool, default=True)  #
    parser.add_argument('--bias', type=bool, default=True)  #
    parser.add_argument('--random_edges_motif', type=int, default=0, help='Node to be explained')

    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
    parser.add_argument('--dropout',type=float, default=0.0, help='dropout rate (1 - keep probability).') ## parser.add_argument('--dropout',type=float, default=0.0, help='dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay',type=float, default=0.0, help='l2 norm') ## 0.0
    parser.add_argument('--hiddens', type=str, default='20-20-20')
    parser.add_argument("--lr", type=float, default=0.001,help='initial learning rate.')
    parser.add_argument('--act', type=str, default='relu', help='activation funciton')  #
    parser.add_argument('--initializer', default='glorot')

    parser.add_argument('--normadj', type=bool, default=False)
    parser.add_argument('--bn', type=bool, default=False)
    parser.add_argument('--concat', type=bool, default=False)
    parser.add_argument('--valid', type=bool, default=False)
    parser.add_argument('--batch', type=bool, default=True)


    parser.add_argument('--save_model',type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='./checkpoints/gcn')

    # ---------------------paramerters for explainers----------------------
    parser.add_argument("--elr", type=float, default=0.05,help='initial learning rate.')
    parser.add_argument('--eepochs', type=int, default=20, help='Number of epochs to train explainer.')


    args, _ = parser.parse_known_args()

    return args

args = get_params()
params = vars(args)

# os.environ["CUDA_VISIBLE_DEVICES"] = args.device
import tensorflow as tf

seed = args.seed
random.seed(args.seed)
np.random.seed(seed)
tf.random.set_seed(seed)

dtype = tf.float32
if args.dtype=='float64':
    dtype = tf.float64

eps = 1e-7
