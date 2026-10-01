#!/bin/bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
D=/scratch/sanchej7/tmp/claude-19646/-nfs-hpc-share-sanchej7-Computer-Vision/cc4dca4e-228f-469e-9df3-2f9b1f7d605a/scratchpad/solutions/turning
cd $D; export MJ_CACHE=$D/mjcache
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
R="arms-turn-turnboth-s1 arms-turn-turncmd-s0 arms-turn-turnhip-s0 arms-dr1.0-s0"
nice -n 10 $PY mj_probe2.py --runs $R --seeds 0 --wz 0.6 -0.6 --noise 1.0 --noise-until 3.0 --tail 4 --out exp4a_until3.json 2>&1 | grep -v "^\[" > exp4a_until3.log
nice -n 10 $PY mj_probe2.py --runs $R --seeds 0 --wz 0.6 -0.6 --noise 1.0 --noise-until 4.0 --tail 4 --out exp4b_until4.json 2>&1 | grep -v "^\[" > exp4b_until4.log
nice -n 10 $PY mj_probe2.py --runs arms-turn-turnboth-s1 --iters 2000 4000 --seeds 0 --wz 0.6 -0.6 --noise 1.0 --out exp4c_evol_noisy.json 2>&1 | grep -v "^\[" > exp4c_evol_noisy.log
echo ALLDONE >> exp4c_evol_noisy.log
