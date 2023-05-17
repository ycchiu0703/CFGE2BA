#!/bin/bash

declare -a path="data_testing"      ## declare -a path="data"
declare -a hiddens="1024-512-128"
declare -a model_name="05-11_13:04:22-GCNClassifier_"      ## model_name="classifier_lynxv2_"  # for classifier
declare -a data_name="5%_poison_connlabcfg"    ## data_name="yancfg_complete_sample2"
declare -a expname="ep300_b32_elr00001_"

# will always run the experiment in first available gpu: 0
CUDA_VISIBLE_DEVICES=0 python exp_interpret_graphs.py $path $hiddens $model_name $data_name $expname > ./trace/trace_interpret_graphs.txt 2>&1 &
