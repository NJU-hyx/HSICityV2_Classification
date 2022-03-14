export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=0
export CUDA_VISIBLE_DEVICES=0,1,

python -m torch.distributed.launch --nproc_per_node=2 tools/train.py