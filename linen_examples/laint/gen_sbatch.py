import argparse
import os.path as osp

parser = argparse.ArgumentParser()

parser.add_argument('--dump-dir', '-du', type=str, default='dumps',
    help='The dump dir for all sweeping exps')
parser.add_argument('--output', '-o', type=str, default='bash/sweep.sh',
    help='The output file name of the script')
parser.add_argument('--logs-dir', '-ld', type=str, default='/scratch/hdd001/home/ywu/LaintResults/sbatch_logs',
    help='The dir to place logs for each run')
parser.add_argument('--batch_size', '-bs', nargs='+', type=int, default=[6250],
    help='The batch size of the experiment')
parser.add_argument('--lrs', '-lrs', nargs='+', type=float, default=[0.0007],
    help='The learning rates of the experiment')
parser.add_argument('--max_length', '-ml', nargs='+', type=int, default=[310],
    help='The max number of tokens of the experiment')
parser.add_argument('--datadir', '-d', type=str, default='/scratch/ssd001/home/ywu/Documents/Isar/PretrainIsar/IsarStepAsciiChar',
    help='The data path of the experiment')
parser.add_argument('--extra-cmd', '-ex', type=str, default=None,
    help='The extra cmd used')
parser.add_argument('--parallel-runs', '-pr', type=int, default=50,
    help='The number of parallel runs')
parser.add_argument('--gpu-type', '-gt', type=str, default='gpu',
    help='The resource type')
parser.add_argument('--qos-type', '-qt', type=str, default='normal',
    help='The resource type')
parser.add_argument('--memory', '-mem', type=int, default=120,
    help='The size of memory to be used')
parser.add_argument('--num-gpus', '-ng', type=int, default=1,
    help='The number of gpus to be used')
parser.add_argument('--num-cpus', '-nc', type=int, default=16,
    help='The number of cpus to be used')
parser.add_argument('--num-samples', '-sample', type=int, default=None,
    help='Randomly sample args multi-times instead of sweeping when not None')

args = parser.parse_args()

def write_cmds(f, cmds):
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --partition={}\n'.format(args.gpu_type))
    f.write('#SBATCH --qos={}\n'.format(args.qos_type))
    f.write('#SBATCH --mem={}G\n'.format(args.memory))
    f.write('#SBATCH --gres=gpu:{}\n'.format(args.num_gpus))
    f.write('#SBATCH --cpus-per-task={}\n'.format(args.num_cpus))
    f.write('#SBATCH --array=0-{}%{}\n'.format(
        len(cmds) - 1, args.parallel_runs))
    log_file = '{}-%A_%a.log'.format('sweep')
    if args.logs_dir is not None:
        log_file = osp.join(args.logs_dir, log_file)
    f.write('#SBATCH --output={}\n'.format(log_file))
    f.write('export XLA_FLAGS=--xla_gpu_cuda_data_dir=/pkgs/cuda-10.1\n')
    f.write('list=(\n')
    for i, cmd in enumerate(cmds):
        f.write('  "{}"\n'.format(cmd))
    f.write(')\n')
    f.write('${list[SLURM_ARRAY_TASK_ID]}')
    # NOTE: comment this one to use self-managed dir for checkpoints
    f.write(' --workdir /checkpoint/${SLURM_JOB_USER}/${SLURM_JOB_ID}\n')

def main():
    cmds = []
    with open(args.output, 'w') as f:
        write_cmds(f, ['python main.py --config configs/default.py'])

if __name__ == "__main__":
    main()
