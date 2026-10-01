#!/bin/bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
D=/scratch/sanchej7/tmp/claude-19646/-nfs-hpc-share-sanchej7-Computer-Vision/cc4dca4e-228f-469e-9df3-2f9b1f7d605a/scratchpad/solutions/turning
cd $D; export MJ_CACHE=$D/mjcache
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
nice -n 10 $PY mj_probe2.py --runs arms-turn-turnboth-s1 arms-turn-turncmd-s0 --seeds 0 1 --wz 0.6 -0.6 --noise 0.0 --obs-noise 1.0 --out exp5_obsnoise.json 2>&1 | grep -v "^\[" > exp5_obsnoise.log
echo ALLDONE >> exp5_obsnoise.log
