#!/bin/bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
D=/scratch/sanchej7/tmp/claude-19646/-nfs-hpc-share-sanchej7-Computer-Vision/cc4dca4e-228f-469e-9df3-2f9b1f7d605a/scratchpad/solutions/turning
cd $D; export MJ_CACHE=$D/mjcache
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
# (a) noise at zero command: does the noisy policy spin by itself?
nice -n 10 $PY mj_probe.py --runs arms-turn-turncmd-s0 arms-turn-turnboth-s1 --seeds 0 1 --wz 0.0 --noise 1.0 --out exp2a_zero.json 2>&1 | grep -v "^\[" > exp2a_zero.log
# (b) dose response and joint subsets
nice -n 10 $PY mj_probe.py --runs arms-turn-turncmd-s0 arms-turn-turnboth-s1 --seeds 0 --wz 0.6 -0.6 --noise 0.25 0.5 --out exp2b_dose.json 2>&1 | grep -v "^\[" > exp2b_dose.log
nice -n 10 $PY mj_probe.py --runs arms-turn-turncmd-s0 arms-turn-turnboth-s1 --seeds 0 --wz 0.6 -0.6 --noise 1.0 --noise-joints legs --out exp2c_legs.json 2>&1 | grep -v "^\[" > exp2c_legs.log
nice -n 10 $PY mj_probe.py --runs arms-turn-turncmd-s0 arms-turn-turnboth-s1 --seeds 0 --wz 0.6 -0.6 --noise 1.0 --noise-joints arms --out exp2d_arms.json 2>&1 | grep -v "^\[" > exp2d_arms.log
# (c) evolution, deterministic, TurnBoth s0 vs s1
nice -n 10 $PY mj_probe.py --runs arms-turn-turnboth-s0 arms-turn-turnboth-s1 --iters 1000 2000 3000 4000 5000 --seeds 0 --wz 0.6 -0.6 --out exp3_evol.json 2>&1 | grep -v "^\[" > exp3_evol.log
echo ALLDONE >> exp3_evol.log
