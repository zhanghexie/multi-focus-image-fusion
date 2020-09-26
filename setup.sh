#!/bin/bash
cifar="./cifar-10-python.tar.gz"
if [ ! -f "$cifar" ];then
	wget "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
fi
mkdir -p save_net/pixel
mkdir -p save_net/block
mkdir -p data_set/pixel/train_set/0
mkdir -p data_set/pixel/train_set/1
mkdir -p data_set/pixel/test_set/0
mkdir -p data_set/pixel/test_set/1
mkdir -p data_set/block/train_set/0
mkdir -p data_set/block/train_set/1
mkdir -p data_set/block/test_set/0
mkdir -p data_set/block/test_set/1
mkdir -p fusion_result
tar -zxvf ./cifar-10-python.tar.gz
