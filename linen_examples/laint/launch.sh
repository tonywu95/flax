DATE=dec14
DUMP_DIR=/scratch/hdd001/home/ywu/LaintResults/Dec14_Flax
python gen_sbatch.py -lr 0.0003 0.0007 -du $DUMP_DIR -gt t4v2 -o bash/${DATE}_lean_flax.sh
vi bash/${DATE}_lean_flax.sh
sbatch bash/${DATE}_lean_flax.sh

