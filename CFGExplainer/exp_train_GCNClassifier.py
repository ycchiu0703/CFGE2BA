import sys

from util.config import args
from util.models import GCN
from util.metrics import accuracy, softmax_cross_entropy
from util.graphprocessor import YANCFG

import networkx as nx
import tensorflow as tf
import mlflow
import mlflow.tensorflow

from datetime import datetime

from tqdm import tqdm

# ------------------------------    
# function to train GCN model
# ------------------------------


def train_GCNClassifier():
    """ runs training for GCN model on yancfg """
    clip_value_min = -2.0
    clip_value_max = 2.0

    # loading the datasets
    data_loader = YANCFG()
    train, _, num_train_samples = data_loader.load_yancfg_data(args.path, 'padded_train', args.malware_list)
    print('+ loaded train dataset')
    # print("train.shape :", train.shape)
    test, _, _ = data_loader.load_yancfg_data(args.path, 'padded_test', args.malware_list)
    test_batch = test.batch(args.batch_size)
    del test
    print('+ loaded test dataset')

    # creating the model
    model = GCN(input_dim=args.d, output_dim=args.c)
    print('+ model: \n', model)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    
    best_acc = 0.0

    # running the training epochs
    for epoch in tqdm(range(args.epochs), disable=args.disable_tqdm):

        # run minibatch training for each epoch
        outputs, labels, losses = [], [], []
        
        train_batch = train.shuffle(num_train_samples).batch(args.batch_size) ## train_batch = train.shuffle(args.batch_size).batch(args.batch_size)
        for batch_id, ts_batch in enumerate(train_batch):
            
            ## clear_session
            tf.keras.backend.clear_session()
            # print('ep: ', epoch, ' batch: ', batch_id)
            
            with tf.device('/gpu:0'):  
                batch_adjs, batch_feats, batch_labels, batch_ids, batch_masks = ts_batch
                
                # ## check shaffle
                # print('ep: ', epoch, ' batch: ', batch_id)
                # print(batch_labels)

                with tf.GradientTape() as tape:
                    output = model.call((batch_feats, batch_adjs), training=True)
                    cross_loss = softmax_cross_entropy(output, batch_labels)
                    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
                    loss = cross_loss + args.weight_decay * lossL2
                    grads = tape.gradient(loss, model.trainable_variables)
                    cliped_grads = [tf.clip_by_value(t, clip_value_min, clip_value_max) for t in grads]
                optimizer.apply_gradients(zip(cliped_grads, model.trainable_variables))
                with tf.device('CPU'):
                    outputs.append(output)
                    labels.append(batch_labels)
                    losses.append(cross_loss)

        output = tf.concat(outputs, axis=0)  # will be of length = #tot-train-samples
        label = tf.concat(labels, axis=0)  # will be same shape as output
        train_loss = tf.reduce_mean(losses, axis=0)  # gets the mean loss for all batches
        train_acc = accuracy(output, label)
        print("ep: ", epoch, "train loss: ", "{:.7f}".format(train_loss), "train acc: ", "{:.7f}".format(train_acc))
    
        # test iterations: will be done per epoch
        # test_batch = test.batch(args.batch_size)  # no need to shuffle
        results = evaluate_model(model, test_batch)

        ## mlflow 
        mlflow.log_metric("train_acc", train_acc.numpy(), step = epoch)
        mlflow.log_metric("train_loss", train_loss.numpy(), step = epoch)
        mlflow.log_metric("test_acc", results['accuracy'].numpy(), step = epoch)
        mlflow.log_metric("test_loss", results['loss'].numpy(), step = epoch)

        if (epoch % args.save_thresh == 0) or (epoch == args.epochs - 1):
            if args.save_model and best_acc <= results['accuracy'].numpy():
                best_acc = results['accuracy'].numpy()
                mlflow.log_metric('Save_model_Train_acc', train_acc, step = epoch)
                mlflow.log_metric('Save_model_Test_acc', best_acc, step = epoch)
                model.save_weights(args.save_path + args.dataset)
                      
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
            val_cross_loss = softmax_cross_entropy(output, batch_labels)
            outputs.append(output)
            labels.append(batch_labels)
            losses.append(val_cross_loss)
    # compute the results
    all_outputs = tf.concat(outputs, axis=0)
    all_labels = tf.concat(labels, axis=0)
    loss = tf.reduce_mean(losses, axis=0)
    acc = accuracy(all_outputs, all_labels)
    # use in dictionary
    results = {
        "loss": loss,
        "accuracy": acc
    }
    return results


# -----------------------------
# Main function
# -----------------------------

def main(arguments):
    """
    Main function
    ----------------
    Args:
    arguments: the sys.args for running code
    """

    # other arguments are left intact as defaults, check config.py
    # add new arguments: model
    args.d = 8     ## args.d = 13 (input dim)
    args.c = 2     ## args.c = 12 (output dim)
    args.n = 512
    args.batch_size = int(arguments[0])  # batch size
    args.path = str(arguments[1])  # the path to load the data
    args.hiddens = str(arguments[2])  # '1024-512-128'
    args.lr = float(arguments[3])  # 0.00001
    args.model_name_flag = str(arguments[4])  # "GCNClassifier"
    
    ## use current datetime to name mlflow model name
    current_datetime = datetime.now().strftime("%m-%d_%H:%M:%S")
    args.save_path = './checkpoints/' + current_datetime + '-' + args.model_name_flag + '_'
    args.dataset = str(arguments[5])  # "connlabcfg"
    args.epochs = int(arguments[6])  # 1000
    args.embnormlize = False  # keep this False: else the output becomes NaN
    args.save_model = True

    # add arguments: for logging results
    # args.writer_path = None  # wont change ##'./logs/classifier/'
    args.disable_tqdm = True  # make False to see progress bar
    args.save_thresh = 5  # save model state every 5 epochs

    args.malware_list = {
        'Benign': 0,
        'Malware': 1
    }

    ## mlflow
    mlflow.set_experiment(args.model_name_flag)
    mlflow.start_run(run_name = "5%_Poison_GCNClassifier_trigger_11-8_random") ## mlflow.start_run(run_name = "5%_Poison_GCNClassifier")
    mlflow.log_param('training_size', 8000)
    mlflow.log_param('testing_size', 2000)   
    mlflow.log_param('dataset', args.dataset)  
    mlflow.log_param('Batch_Size', args.batch_size)
    mlflow.log_param('Learning_Rate', args.lr)
    mlflow.log_param('Epochs', args.epochs)
    mlflow.log_param('Save_thresh', args.save_thresh)
    mlflow.log_param('input_dim', args.d)
    mlflow.log_param('output_dim', args.c)
    mlflow.log_param('dropout_rate', args.dropout)
    mlflow.log_param('hiddens', args.hiddens)
    mlflow.log_param('datetime', current_datetime)


    

    # debugging argument
    args.debug = False  # prints out the data loading step + loads only 1 graph per sample ## args.debug = False
    if args.debug:
        print("Experimenting in DEBUG mode!")
    
    # run train()
    train_GCNClassifier()
    
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
