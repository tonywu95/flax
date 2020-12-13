export XLA_FLAGS=--xla_gpu_cuda_data_dir=/pkgs/cuda-10.1
rm -rf test
python main.py --workdir=./test --config configs/default.py 
#python main.py --workdir=./test --config configs/debug.py 
#CUDA_VISIBLE_DEVICES=0 python main.py --workdir=./test --config configs/debug.py 
