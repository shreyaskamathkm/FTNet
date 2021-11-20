#!/bin/sh
##https://help.rc.ufl.edu/doc/Using_Variables_in_SLURM_Jobs
#SBATCH -N 1
#SBATCH --mem=32G  # memory in Mb
#SBATCH --cpus-per-task=16
#SBATCH -p preempt
#SBATCH --requeue
#SBATCH -t 7-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --signal=SIGUSR1@90
#SBATCH -e error_%j_%x.dat   # send stderr to errfile
#SBATCH -o output_%j_%x.dat # send stdout to outfile


module purge
module load singularity/3.6.1

module load cuda/10.2
module load cudnn/7.1
container=eclair_pyt_1_6_0_pl_1_3_4_cuda_10_2_ubuntu_18_04.sif

export SINGULARITY_BINDPATH="./cluster/:/workdir"

MODEL=ftnet
BACKBONE=${BACKBONE}
GPUS=2
FILTERS=128
EDGES='3'
NBLOCKS=2
NODES=1
ALPHA=${ALPHA} 

DATASET=cityscapes_thermal_combine
RUNDIR=./../../../Training_Paper/Lightning/"$MODEL"_"$BACKBONE"_"$FILTERS"_"$EDGES"_"$NBLOCKS"/"$DATASET"/
mkdir -p "$RUNDIR"

: '
## Pretraining
singularity exec --nv --pwd /workdir/FTNet/Codes/src/bash/ /cluster/singularity/"$container"  \
python  ./../lightning_scripts/main.py \
--mode 'train' \
--no-of-filters "$FILTERS" \
--edge-extracts "$EDGES" \
--loss-weight "$ALPHA" \
--num-blocks "$NBLOCKS" \
--train_only True \
--model "$MODEL" \
--pretrained-base True \
--backbone "$BACKBONE" \
--dataset "$DATASET" \
--dataset-path './../../../../Thermal_Segmentation/Dataset/' \
--base-size '520'  \
--crop-size '480' \
--train-batch-size 16 \
--val-batch-size 16 \
--epochs 100 \
--optimizer 'SGD' \
--lr 0.01 \
--scheduler-type 'poly_warmstartup' \
--warmup-iters 0 \
--warmup-factor 0.3333 \
--warmup-method 'linear' \
--save-images-as-subplots False \
--save-images False \
--debug False \
--workers 16 \
--momentum 0.9 \
--weight-decay 0.0001 \
--beta1 0.9 \
--beta2 0.999 \
--epsilon 1e-8 \
--seed 0 \
--gpus $GPUS \
--num-nodes $NODES \
--distributed-backend 'ddp' \
--wandb-name-ext '' \
--save-dir "$RUNDIR" >"$RUNDIR"/log_train_cityscape.txt
'
## FineTuning
for DATASET2 in 'soda' 'mfn' 'scutseg'
do

RUNDIR2=./../../../Training_Paper/Lightning/"$MODEL"_"$BACKBONE"_"$FILTERS"_"$EDGES"_"$NBLOCKS"_"$MSV"/"$DATASET2"_Alpha_"$ALPHA"/
mkdir -p "$RUNDIR2"


singularity exec --nv --pwd /workdir/FTNet/Codes/src/bash/ /cluster/singularity/"$container"  \
python  ./../lightning_scripts/main.py \
--mode 'train' \
--train_only False \
--no-of-filters "$FILTERS" \
--num-blocks "$NBLOCKS" \
--edge-extracts "$EDGES" \
--loss-weight "$ALPHA" \
--model "$MODEL" \
--pretrain-checkpoint "$RUNDIR"/ckpt/last.ckpt \
--pretrained-base False \
--backbone "$BACKBONE" \
--dataset "$DATASET2" \
--dataset-path './../../../../Thermal_Segmentation/Dataset/' \
--base-size '520'  \
--crop-size '480' \
--train-batch-size 16 \
--val-batch-size 16 \
--test-batch-size 1 \
--epochs 100 \
--optimizer 'SGD' \
--lr 0.001 \
--scheduler-type 'poly_warmstartup' \
--warmup-iters 0 \
--warmup-factor 0.3333 \
--warmup-method 'linear' \
--save-images-as-subplots False \
--save-images False \
--debug False \
--workers 16 \
--momentum 0.9 \
--weight-decay 0.0001 \
--beta1 0.9 \
--beta2 0.999 \
--epsilon 1e-8 \
--seed 0 \
--gpus $GPUS \
--num-nodes $NODES \
--distributed-backend 'ddp' \
--wandb-name-ext '' \
--save-dir "$RUNDIR2" >"$RUNDIR2"/log_train_"$DATASET2"_Alpha_"$ALPHA".txt


## Testing last ckpt

singularity exec --nv --pwd /workdir/FTNet/Codes/src/bash/ /cluster/singularity/"$container"  \
python  ./../lightning_scripts/main.py \
--mode 'test' \
--model "$MODEL" \
--edge-extracts "$EDGES" \
--loss-weight "$ALPHA" \
--num-blocks "$NBLOCKS" \
--backbone "$BACKBONE" \
--no-of-filters "$FILTERS" \
--test-checkpoint "$RUNDIR2"/ckpt/last.ckpt \
--pretrained-base False \
--dataset "$DATASET2" \
--dataset-path './../../../../Thermal_Segmentation/Dataset/' \
--test-batch-size 1 \
--save-images True \
--save-images-as-subplots False \
--debug False \
--workers 16 \
--seed 0 \
--gpus 1 \
--num-nodes 1 \
--distributed-backend 'dp' \
--wandb-name-ext '' \
--save-dir "$RUNDIR2"/Last_CKPT/ >"$RUNDIR2"/log_last_test_"$DATASET2"_Alpha_"$ALPHA".txt


## Testing
singularity exec --nv --pwd /workdir/FTNet/Codes/src/bash/ /cluster/singularity/"$container"  \
python  ./../lightning_scripts/main.py \
--mode 'test' \
--model "$MODEL" \
--edge-extracts "$EDGES" \
--loss-weight "$ALPHA" \
--num-blocks "$NBLOCKS" \
--backbone "$BACKBONE" \
--no-of-filters "$FILTERS" \
--test-monitor 'val_mIOU' \
--test-monitor-path "$RUNDIR2"/ckpt/ \
--pretrained-base False \
--dataset "$DATASET2" \
--dataset-path './../../../../Thermal_Segmentation/Dataset/' \
--test-batch-size 1 \
--save-images True \
--save-images-as-subplots False \
--debug False \
--workers 16 \
--seed 0 \
--gpus 1 \
--num-nodes 1 \
--distributed-backend 'dp' \
--wandb-name-ext '' \
--save-dir "$RUNDIR2"/Best_MIOU/ >"$RUNDIR2"/log_best_test_"$DATASET2"_Alpha_"$ALPHA".txt


done
