#!/bin/bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
D=/scratch/sanchej7/tmp/claude-19646/-nfs-hpc-share-sanchej7-Computer-Vision/cc4dca4e-228f-469e-9df3-2f9b1f7d605a/scratchpad/solutions/turning
cd $D; export MJ_CACHE=$D/mjcache
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
R="arms-turn-turnboth-s1 arms-turn-turncmd-s0"
OBS_NOISE_MASK=all nice -n 10 $PY mj_probe2.py --runs $R --seeds 0 --wz 0.6 -0.6 --obs-noise 0.25 --out exp6a_obs025.json 2>&1 | grep -v "^\[" > exp6a_obs025.log
OBS_NOISE_MASK=jv nice -n 10 $PY mj_probe2.py --runs $R --seeds 0 --wz 0.6 -0.6 --obs-noise 1.0 --out exp6b_jvonly.json 2>&1 | grep -v "^\[" > exp6b_jvonly.log
OBS_NOISE_MASK=nojv nice -n 10 $PY mj_probe2.py --runs $R --seeds 0 --wz 0.6 -0.6 --obs-noise 1.0 --out exp6c_nojv.json 2>&1 | grep -v "^\[" > exp6c_nojv.log
echo ALLDONE >> exp6c_nojv.log
