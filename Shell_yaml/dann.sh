#!/usr/bin/env bash
GPU_ID=0
CQDXGear_data_dir=../../Dataset/CQDX/CQDXGear/gear_fault_rawdata_1024000x1_simple.mat
CQDXBear_data_dir=../../Dataset/CQDX/CQDXBear/ZouBearingFaultRawConstantLoad.mat
CWRUBear_data_dir=../../Dataset/For_the_strange_Benchmark/CWRU
CWRUBear48K_data_dir=../../Dataset/For_the_strange_Benchmark/CWRU48K

#CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXGear1DaFFT --data_dir $CQDXGear_data_dir --src_wc 0 --tgt_wc 1 | tee Log/DANN_Log/DANN_cqdxgear1d021.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXGear1DaFFT --data_dir $CQDXGear_data_dir --src_wc 0 --tgt_wc 2 | tee Log/DANN_Log/DANN_cqdxgear1d022.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXGear1DaFFT --data_dir $CQDXGear_data_dir --src_wc 0 --tgt_wc 3 | tee Log/DANN_Log/DANN_cqdxgear1d023.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXGear1DaFFT --data_dir $CQDXGear_data_dir --src_wc 1 --tgt_wc 0 | tee Log/DANN_Log/DANN_cqdxgear1d120.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXGear1DaFFT --data_dir $CQDXGear_data_dir --src_wc 1 --tgt_wc 2 | tee Log/DANN_Log/DANN_cqdxgear1d122.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXGear1DaFFT --data_dir $CQDXGear_data_dir --src_wc 1 --tgt_wc 3 | tee Log/DANN_Log/DANN_cqdxgear1d123.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXGear1DaFFT --data_dir $CQDXGear_data_dir --src_wc 2 --tgt_wc 0 | tee Log/DANN_Log/DANN_cqdxgear1d220.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXGear1DaFFT --data_dir $CQDXGear_data_dir --src_wc 2 --tgt_wc 1 | tee Log/DANN_Log/DANN_cqdxgear1d221.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXGear1DaFFT --data_dir $CQDXGear_data_dir --src_wc 2 --tgt_wc 3 | tee Log/DANN_Log/DANN_cqdxgear1d223.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXGear1DaFFT --data_dir $CQDXGear_data_dir --src_wc 3 --tgt_wc 0 | tee Log/DANN_Log/DANN_cqdxgear1d320.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXGear1DaFFT --data_dir $CQDXGear_data_dir --src_wc 3 --tgt_wc 1 | tee Log/DANN_Log/DANN_cqdxgear1d321.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXGear1DaFFT --data_dir $CQDXGear_data_dir --src_wc 3 --tgt_wc 2 | tee Log/DANN_Log/DANN_cqdxgear1d322.log

##########------------##########
# remote contrast experiments:
##########------------##########

##############################################################################################################
############################################# CQDXBear #######################################################
##############################################################################################################
# # mmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss mmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss mmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss mmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss mmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss mmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss mmd
# # adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss adv
# # lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss lmmd
# # coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss coral
# # bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss bnm
# # mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss mmd_adv
# # lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss lmmd_adv
# # plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss plmmd_adv
# # source only
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBearFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss sourceOnly

##############################################################################################################
############################################# CWRUBear #######################################################
##############################################################################################################
# mmd:
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --transfer_loss mmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --transfer_loss mmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --transfer_loss mmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --transfer_loss mmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --transfer_loss mmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --transfer_loss mmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --transfer_loss mmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --transfer_loss mmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --transfer_loss mmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --transfer_loss mmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --transfer_loss mmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --transfer_loss mmd
# # adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --transfer_loss adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --transfer_loss adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --transfer_loss adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --transfer_loss adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --transfer_loss adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --transfer_loss adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --transfer_loss adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --transfer_loss adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --transfer_loss adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --transfer_loss adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --transfer_loss adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --transfer_loss adv
# # lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --transfer_loss lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --transfer_loss lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --transfer_loss lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --transfer_loss lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --transfer_loss lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --transfer_loss lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --transfer_loss lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --transfer_loss lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --transfer_loss lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --transfer_loss lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --transfer_loss lmmd
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --transfer_loss lmmd
# # coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --transfer_loss coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --transfer_loss coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --transfer_loss coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --transfer_loss coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --transfer_loss coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --transfer_loss coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --transfer_loss coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --transfer_loss coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --transfer_loss coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --transfer_loss coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --transfer_loss coral
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --transfer_loss coral
# # bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --transfer_loss bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --transfer_loss bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --transfer_loss bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --transfer_loss bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --transfer_loss bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --transfer_loss bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --transfer_loss bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --transfer_loss bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --transfer_loss bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --transfer_loss bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --transfer_loss bnm
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --transfer_loss bnm
# # mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --transfer_loss mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --transfer_loss mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --transfer_loss mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --transfer_loss mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --transfer_loss mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --transfer_loss mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --transfer_loss mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --transfer_loss mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --transfer_loss mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --transfer_loss mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --transfer_loss mmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --transfer_loss mmd_adv
# # lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --transfer_loss lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --transfer_loss lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --transfer_loss lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --transfer_loss lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --transfer_loss lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --transfer_loss lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --transfer_loss lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --transfer_loss lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --transfer_loss lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --transfer_loss lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --transfer_loss lmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --transfer_loss lmmd_adv
# # plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --transfer_loss plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --transfer_loss plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --transfer_loss plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --transfer_loss plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --transfer_loss plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --transfer_loss plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --transfer_loss plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --transfer_loss plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --transfer_loss plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --transfer_loss plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --transfer_loss plmmd_adv
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --transfer_loss plmmd_adv


###########################################################################
############################## Source Only ################################
###########################################################################
# # CQDXBear
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1DaFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1DaFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1DaFFT --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1DaFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1DaFFT --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1DaFFT --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --transfer_loss sourceOnly
# # CWRUBear
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 1 --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 2 --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 0 --tgt_wc 3 --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 0 --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 2 --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 1 --tgt_wc 3 --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 0 --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 1 --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 2 --tgt_wc 3 --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 0 --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 1 --transfer_loss sourceOnly
# CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer.py --config Shell_yaml/dann.yaml --num_class 10 --dataset CWRUBear1DaFFT --data_dir $CWRUBear_data_dir --src_wc 3 --tgt_wc 2 --transfer_loss sourceOnly


############################################################################################
############################## Single cnn1d for CQDXBear ###################################
############################################################################################
# mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss mmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss mmd
# adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss adv
lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss lmmd
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss lmmd
# coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss coral
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss coral
# bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss bnm
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss bnm
# mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss mmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss mmd_adv
# lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss lmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss lmmd_adv
# plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss plmmd_adv
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss plmmd_adv
# source only
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss sourceOnly
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 0 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss sourceOnly
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss sourceOnly
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 1 --tgt_wc 2 --acc_preview Best_acc.txt --transfer_loss sourceOnly
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 0 --acc_preview Best_acc.txt --transfer_loss sourceOnly
CUDA_VISIBLE_DEVICES=$GPU_ID python train_transfer_tmp.py --config Shell_yaml/dann.yaml --num_class 5 --dataset CQDXBear1D --data_dir $CQDXBear_data_dir --src_wc 2 --tgt_wc 1 --acc_preview Best_acc.txt --transfer_loss sourceOnly
