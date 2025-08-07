#!/bin/bash
####### multi machines multi gpu cards
export NCCL_DEBUG=info
export NCCL_SOCKET_IFNAME=eth0

MASTER_IP=$(submit/get_master_ip.sh)
echo ${MASTER_IP}
RANK=$1
echo $1

if [ ${RANK} -eq 0 ]
then
 export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_4,mlx5_5,mlx5_6
 echo "export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_4,mlx5_5,mlx5_6"

else
echo "not master."
fi

export OMP_NUM_THREADS=4

torchrun --nnodes=4 --nproc_per_node=8 --master_port 4880 --node_rank ${RANK} --master_addr ${MASTER_IP}  pretrain_oneref_with_mrefm_multi_node.py




