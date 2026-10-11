import json,os,pathlib,subprocess,sys
p=json.loads(pathlib.Path('/nfs/stak/users/sanchej7/humanoid-native-20261009/orb-build-v5/step-launch-plan-21739893.json').read_text())
env={k:v for k,v in os.environ.items() if not k.startswith(("SLURM_","APPTAINERENV_"))}
env["TMPDIR"]="/tmp"
sys.exit(subprocess.run(p["command"],env=env).returncode)
