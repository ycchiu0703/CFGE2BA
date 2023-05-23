#!/bin/bash

declare -a batch_size="10"                                  ## batch_size="10"
declare -a path="Partition_Trigger_samples"                              ## path="data"
declare -a hiddens="1024-512-128"                           ## hiddens="1024-512-128"
declare -a lr="0.0001"
declare -a model_name="GCNClassifier_"                      ## model_name="GCNClassifier_"      ## model_name="05-11_13:04:22-GCNClassifier_"
declare -a data_name="connlabcfg"                           ## data_name="connlabcfg"           ## data_name="5%_poison_connlabcfg" 
declare -a epochs="50"

CUDA_VISIBLE_DEVICES=0 python Test.py $batch_size $path $hiddens $lr $model_name $data_name $epochs > ./trace/testing_sample2_classifier.txt 2>&1 &

