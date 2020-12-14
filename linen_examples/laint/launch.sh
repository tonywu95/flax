DATE=dec13
DUMP_DIR=/scratch/hdd001/home/ywu/LaintResults/Dec13_Flax
python gen_sbatch.py -lr 0.02 0.05 -du $DUMP_DIR -gt t4v2 -o bash/${DATE}_lean_flax.sh
vi bash/${DATE}_lean_flax.sh
sbatch bash/${DATE}_lean_flax.sh

