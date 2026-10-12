import pathlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=pathlib.Path(__file__).resolve().parent
full=root/'full/native';result=json.loads((full/'result.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,6),sharex=True,layout='constrained')
for i,mode in enumerate(['stereo','stereo_inertial']):
 rows=[json.loads(x) for x in (full/mode/'frames.jsonl').read_text().splitlines()]
 t=np.array([x['native']['timestamp_s'] for x in rows]);track=np.array([x['native']['tracked'] for x in rows]);accepted=np.array([x['native']['tracked'] and (mode=='stereo' or x['native']['inertial']['map_imu_initialized']) for x in rows])
 a=axes[i,0];a.step(t,track.astype(int),label='Native tracking',where='post',color='#2878b5',lw=1.5)
 if mode!='stereo':a.step(t,accepted.astype(int),label='Tracking + IMU initialized',where='post',color='#c65c15',lw=1.5)
 a.set_yticks([0,1],['No','Yes']);a.set_ylim(-.1,1.15);a.set_title(('Stereo' if mode=='stereo' else 'Stereo–IMU')+f': {sum(accepted):,}/1,200 initialized tracked frames',loc='left',fontsize=11);a.legend(loc='lower right',fontsize=8)
 a=axes[i,1]
 for key,color,label in [('pipeline_wall_s','#2878b5','Read + rectification + IPC + native'),('compute_seconds','#c65c15','Native TrackStereo')]:
  v=np.array([x['native'][key] if key=='compute_seconds' else x['timing'][key] for x in rows])*1000
  a.plot(t,v,linewidth=.75,alpha=.8,color=color,label=f'{label}; p95 {np.quantile(v,.95):.1f} ms')
 a.axhline(50,ls=':',color='#555555',lw=1,label='20 Hz sensor interval (50 ms)');a.set_ylabel('Offline elapsed time per frame (ms)');a.set_ylim(bottom=0);a.legend(loc='upper right',fontsize=8)
for a in axes.ravel():a.set_xlim(0,60);a.grid(alpha=.18)
for a in axes[-1]:a.set_xlabel('Original sensor time from first stereo pair (s)')
fig.suptitle('EuRoC V1_01_easy: fixed 60-second recorded-data replay',fontsize=15)
fig.supxlabel('Offline, unpaced CPU replay · no ground truth used for inference · external recording, not validation of the physical robot',fontsize=9)
for ext in ['png','svg']:fig.savefig(root/f'euroc-tracking-latency.{ext}',dpi=160)
