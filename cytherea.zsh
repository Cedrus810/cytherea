#!/bin/zsh
#PBS -q default
#PBS -l select=1:ncpus=32:ngpus=1:mem=60gb:host=yayoi41
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -N cytherea
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH
# openfe_env 由 run.zsh 自己激活（NFS 上那一份，conda activate 可能要几分钟，日志缓冲，别当成卡死）

cd /home/ruigengji/cytherea
export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh && mamba activate openmm_dev
cd /home/ruigengji/cytherea
export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh && mamba activate openmm_dev
python examples/alanine_dipeptide/ref_long.py \
  --out /home/ruigengji/cytherea/runs/ala2_ref --total-ns 1000 --seed 20261001 \
  --platform CUDA --system-dir /home/ruigengji/cytherea/runs/ala2_system --resume \
  2>&1 | tee runs/ala2_ref/resume_$(date +%Y%m%d_%H%M).log
