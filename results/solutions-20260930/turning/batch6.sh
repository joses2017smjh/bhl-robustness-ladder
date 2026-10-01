#!/bin/bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
D=/scratch/sanchej7/tmp/claude-19646/-nfs-hpc-share-sanchej7-Computer-Vision/cc4dca4e-228f-469e-9df3-2f9b1f7d605a/scratchpad/solutions/turning
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder; U=$REPO/external/Berkeley-Humanoid-Lite
cd $D; export PYTHONPATH=$REPO/src:$REPO/scripts
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
for spec in "arms-turn-turnboth-s0 1.0" "arms-turn-turnboth-s1 1.0" "arms-turn-turnboth-s1 0.0"; do
  set -- $spec; RUN=$1; M=$2
  DEP=$(ls -d $U/logs/rsl_rl/humanoid/*_$RUN | sort | tail -1)/exported/deploy.yaml
  date; nice -n 10 $PY push_obsnoise.py $M --deploy-cfg $DEP --upstream $U --cache-dir $D/mjcache/push --out $D/push_${RUN}_obs${M}.csv --label $RUN --variant humanoid --episode-s 12 --seeds 10 --push-speed 0.5 --terrain-difficulty 0 2>&1 | grep -v "^\[" | tail -3
  $PY -c "import csv,sys; r=list(csv.DictReader(open(sys.argv[1]))); print(sys.argv[1], sum(x['fell']=='True' for x in r), '/', len(r))" $D/push_${RUN}_obs${M}.csv
done
echo ALLDONE
