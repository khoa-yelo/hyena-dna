#!/bin/bash
#SBATCH -p gpu,owners
#SBATCH --time=01:00:00                                     # how much time to run
#SBATCH --mem=32G                                           # how much mem in MBs
#SBATCH -C "GPU_CC:7.0|GPU_CC:7.5|GPU_CC:8.0|GPU_CC:8.6"    # the | logic for any of the features listed. 
#SBATCH -G 1
#SBATCH --job-name=splash-hyena-dna                                # name the job jupyter host
#SBATCH --output=output.%x.%A_%a
#SBATCH --error=error.%x.%A_%a

HYE_REPO=/scratch/users/khoang99/repos/hyena-dna
HYE_IMG=/scratch/users/khoang99/envs/hyena-dna-nt6_latest.sif
HYE="singularity run --nv --writable-tmpfs -B $HYE_REPO $HYE_IMG"
FASTA_FILE="$HYE_REPO/data/splash/RE_CTGCAG_pSpectral_lt_01.fasta"
LABEL_FILE=""
$HYE python -m train wandb=null experiment=splash/splash_pretrain dataset.fasta_file="$FASTA_FILE" dataset.label_file="$LABEL_FILE"
echo "Training done"

# Obtain last checkpoint (copied from zyzhang code)
checkpoint_path=outputs
most_recent_dir=$(ls ${checkpoint_path} -lt | grep '^d' | head -1 | awk '{print $9}')
checkpoint_path=${checkpoint_path}/${most_recent_dir}
most_recent_dir=$(ls ${checkpoint_path} -lt | grep '^d' | head -1 | awk '{print $9}')
checkpoint_path=${checkpoint_path}/${most_recent_dir}
checkpoint_path=${checkpoint_path}/checkpoints/last.ckpt
echo "Checkpoint path: $checkpoint_path"
$HYE python "$HYE_REPO/evals/splash_encode.py" --ckpt_path "$checkpoint_path" --input_fasta "$FASTA_FILE" --output_dir "$most_recent_dir/embeddings"