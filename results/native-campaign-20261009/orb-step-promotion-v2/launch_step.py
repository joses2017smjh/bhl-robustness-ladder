import json,os,pathlib,subprocess,sys
p=json.loads(pathlib.Path('/nfs/stak/users/sanchej7/humanoid-native-20261009/orb-step-promotion-v2/step-launch-plan.json').read_text())
completion=json.loads(pathlib.Path("/nfs/stak/users/sanchej7/humanoid-native-20261009/orb-build-v6/orb-native-build-v6-20261009-21739893/completion.json").read_text())
if completion.get("status")!="PASS" or completion.get("exit_status")!=0: raise SystemExit("Genuine successful C++ build required before orchestration")
env={k:v for k,v in os.environ.items() if not k.startswith(("SLURM_","APPTAINERENV_"))}
env["TMPDIR"]="/tmp"
sys.exit(subprocess.run(p["command"],env=env).returncode)
