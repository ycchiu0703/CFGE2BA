import sys
import os
from util.config import args
from util.models import GCN
from util.metrics import accuracy
from util.graphprocessor import YANCFG

import tensorflow as tf
import mlflow
from tqdm import tqdm

from Explainer import ExplainerModule


def test_GCNClassifier():

    model = GCN(input_dim=args.d, output_dim=args.c)    
    model.load_weights(args.save_path + args.dataset)  # load the weights
    print("+ load model weights Successfully!")
    
    data_loader = YANCFG()
    test, _, _ = data_loader.load_yancfg_data(args.path, 'AFCG_test', args.malware_list) ##    test, _, _ = data_loader.load_yancfg_data(args.path, 'padded_test', args.malware_list)

    test_batch = test.batch(args.batch_size)
    del test
    print('+ loaded test dataset')

    results = evaluate_model(model, test_batch)
    print("poison accuracy: ", results["accuracy"].numpy())
    print("ASR: ", 1 - results["accuracy"].numpy())

    return 

def evaluate_model(model, batch_dataset):
    """
    step through all the data samples in test dataset
    and compute scores for accuracy and validation loss
    ----------------
    Args
    model (tf.model): the intance of the model
    batch_dataset (tf.data.Dataset): a dataset class, batched and shuffled
    ----------------
    Returns
    results (dict): a dictionary for {'loss', 'accuracy'}
    """
    outputs, labels, losses = [], [], []
    # loop through all val. batches
    for batch_id, ts_batch in enumerate(batch_dataset):
        with tf.device('/gpu:0'):   ## with tf.device('/gpu:0'):
            batch_adjs, batch_feats, batch_labels, batch_ids, batch_masks = ts_batch
            output = model.call((batch_feats, batch_adjs), training=False)
            output = tf.nn.softmax(output)
            outputs.append(output)
            labels.append(batch_labels)
    # compute the results
    all_outputs = tf.concat(outputs, axis=0)
    all_labels = tf.concat(labels, axis=0)

    print("num of sample : ", len(all_labels))
    print(tf.argmax(all_outputs, 1).numpy().tolist())
    print(tf.argmax(all_labels, 1).numpy().tolist())

    acc = accuracy(all_outputs, all_labels)
    # print('output :', all_outputs)
    print('acc', acc)

    mlflow.log_param('num of sample', len(all_labels))
    mlflow.log_param('ASR', 1 - acc.numpy())

    # use in dictionary
    results = {
        "accuracy": acc
    }
    return results

def main(arguments):
    """
    Main function
    ----------------
    Args:
    arguments: the sys.args for running code
    """
    # other arguments are left intact as defaults, check config.py
    # add new arguments: model
    args.d = 8      ## args.d = 13 (input dim)
    args.c = 2      ## args.c = 12 (output dim)
    args.n = 512  # the number of nodes in padded graph (fixed for experiment) ## args.n = 4690
    args.batch_size = int(arguments[0])  # batch size
    args.path = str(arguments[1])  # the path to load the data
    args.hiddens = str(arguments[2])  # '1024-512-128'
    args.elr = float(arguments[3])  # [explainer] optimizer learning rate 0.005 (default)
    args.model_name_flag = str(arguments[4])  # 'trial_gcn_'
    args.save_path = './checkpoints/' + args.model_name_flag
    args.dataset = str(arguments[5])  # 'yancfg_test'
    args.eepochs = int(arguments[6])  # [explainer] epochs 1000
    args.embnormlize = False  # keep this False: else the output becomes NaN
    
    args.writer_path = './logs/explainer/'  # wont change
    args.disable_tqdm = True  # make True to hide progress bar
    args.save_thresh = 1  # save model state every 1 epoch

    # debugging argument
    args.debug = False  # prints out the data loading step + loads only 1 graph per sample

    # new params [explainer]
    # args.explainer_name = str(arguments[7])
    # args.explainer_path = './checkpoints/explainer_' + str(arguments[7]) + args.model_name_flag + args.dataset  # path to save the explainer model
    # args.results_save_path = './interpretability_results'  # the path to save the results (add a git-ignore?)
    
    args.malware_list = {
        'Benign': 0,
        'Malware': 1
    }
    ## mlflow
    mlflow.set_experiment("FCG_Test_ASR")
    mlflow.start_run(run_name = "Clean_GCNClassifier")      ## mlflow.start_run(run_name = "Clean_GCNClassifier") ## mlflow.start_run(run_name = "5%_GCNClassifier_trigger_2-1")

    # run explain code
    test_GCNClassifier()
    mlflow.end_run()
    
    return


# running the code
if __name__ == "__main__":
    print("sys.args: ", sys.argv)

    ##  GPU settings
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    main(sys.argv[1:])
