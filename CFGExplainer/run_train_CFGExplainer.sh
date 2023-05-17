#!/bin/bash

declare -a batch_size="32" 
declare -a path="data_poison"                   ## path="data"                  ## path="data_poison"
declare -a hiddens="1024-512-128"
declare -a elr="0.00001" 
declare -a model_name="05-11_13:04:22-GCNClassifier_"      ## model_name="classifier_lynxv2_"
declare -a data_name="5%_poison_connlabcfg"     ## data_name="connlabcfg"       ## data_name="5%_poison_connlabcfg" 
declare -a eepochs="300"
declare -a expname="ep300_b32_elr00001_"

CUDA_VISIBLE_DEVICES=0 python exp_train_CFGExplainer.py $batch_size $path $hiddens $elr $model_name $data_name $eepochs $expname > ./trace/training_sample2_CFGExplainer.txt 2>&1 &
