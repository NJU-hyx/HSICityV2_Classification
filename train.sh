export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=0
export CUDA_VISIBLE_DEVICES=0

# python -m torch.distributed.launch --nproc_per_node=1 tools/train.py
python tools/train.py