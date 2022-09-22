#!/usr/bin/env bash
GPU_ID=6
CQDXGear_data_dir=../../Dataset/CQDX/CQDXGear/gear_fault_rawdata_1024000x1_simple.mat
CQDXBear_data_dir=../../Dataset/CQDX/CQDXBear/ZouBearingFaultRawConstantLoad.mat
CWRUBear_data_dir=../../Dataset/For_the_strange_Benchmark/CWRU


##############################################################################################################
####################################### Single cnn1d for CWRUBear ############################################
##############################################################################################################
# mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss mmd
# adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss adv
# lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss lmmd
# coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss coral
# bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss bnm
# mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss mmd_adv
# lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss lmmd_adv
# plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss plmmd_adv
# source only
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss sourceOnly
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss sourceOnly
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss sourceOnly
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss sourceOnly
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss sourceOnly
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss sourceOnly
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss sourceOnly
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss sourceOnly
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --acc_preview Best_acc_dan.txt --transfer_loss sourceOnly
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --acc_preview Best_acc_dan.txt --transfer_loss sourceOnly
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --acc_preview Best_acc_dan.txt --transfer_loss sourceOnly
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBearFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --acc_preview Best_acc_dan.txt --transfer_loss sourceOnly

