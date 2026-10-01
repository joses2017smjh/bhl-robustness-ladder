#!/bin/bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
D=/scratch/sanchej7/tmp/claude-19646/-nfs-hpc-share-sanchej7-Computer-Vision/cc4dca4e-228f-469e-9df3-2f9b1f7d605a/scratchpad/solutions/turning
cd $D; export MJ_CACHE=$D/mjcache
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
R=arms-turn-turnhip-s0,arms-turn-turnhip-s1,arms-turn-turnhip-s2,arms-turn-turntrack-s0,arms-turn-turntrack-s1,arms-turn-turntrack-s2,arms-turn-turnboth-s1,arms-turn-turnboth-s2,arms-turn-turncmd-s0,arms-turn-turncmd-s1,arms-turn-turncmd-s2,arms-dr1.0-s0,arms-turn-turnboth-s0
nice -n 10 $PY v2_obsnoise.py $R 1.0 2>&1 | grep -v "^\[" > v2_obsnoise_x1.0.log
echo ALLDONE >> v2_obsnoise_x1.0.log
