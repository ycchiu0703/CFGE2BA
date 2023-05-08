#!/bin/bash

declare -a batch_size="10"
declare -a path="data"
declare -a hiddens="1024-512-128" ## hiddens="1024-512-128"
declare -a lr="0.0001"
declare -a model_name="GCNClassifier"  ## model_name="GCNClassifier" ## model_name="GCNClassifier"
declare -a data_name="connlabcfg" ## data_name="5%_Poison_connlabcfg" ## data_name="connlabcfg"
declare -a epochs="200"

CUDA_VISIBLE_DEVICES=0 python exp_train_GCNClassifier.py $batch_size $path $hiddens $lr $model_name $data_name $epochs > ./trace/training_sample2_classifier.txt 2>&1 &

