#!/usr/bin/env python3
"""Plot genuine native features, positive depth matches and the unchanged gate."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from bhl_robust.research.native_orb import sha256,validate_response


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--frames',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.receipt.json').exists():
        raise ValueError('new output paths required')
    figure,axes=plt.subplots(len(args.frames),1,figsize=(10,3*len(args.frames)),squeeze=False,constrained_layout=True)
    sources={}
    for axis,path in zip(axes[:,0],args.frames):
        rows=[]
        for line in path.read_text().splitlines():
            row=json.loads(line);rows.append(validate_response(row,row['timestamp_s']))
        if any('diagnostics' not in row for row in rows):raise ValueError('native telemetry required')
        t=np.array([row['timestamp_s'] for row in rows])
        n=np.array([row['diagnostics']['features_total'] for row in rows])
        depth=np.array([row['diagnostics']['positive_stereo_depth_matches'] for row in rows])
        tracked=np.array([row['tracked'] for row in rows])
        axis.plot(t,n,label='Actual native extracted features',color='#2563eb')
        axis.plot(t,depth,label='Actual positive stereo-depth matches',color='#15803d')
        axis.axhline(500,color='#b91c1c',linestyle='--',label='Initialization requires >500 features')
        axis.fill_between(t,0,1,where=~tracked,transform=axis.get_xaxis_transform(),color='#64748b',alpha=.12)
        name=path.parent.parent.name.replace('_native','').replace('_',' ')
        axis.set_title(f'{name}: {int(tracked.sum())}/{len(rows)} native tracked frames')
        axis.set_ylabel('Native frame count');axis.set_xlabel('Original sensor time (s)')
        axis.set_ylim(bottom=0);axis.grid(alpha=.2)
        sources[str(path)]={'sha256':sha256(path),'frames':len(rows),'tracked':int(tracked.sum())}
    axes[0,0].legend(fontsize=8,loc='upper left')
    figure.suptitle('Read-only ORB-SLAM3 diagnostics on the original consumed capture\nGray shading: native pose unavailable; native algorithm and feature gate unchanged',fontsize=11)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(args.output,dpi=170)
    plt.close(figure)
    args.output.with_suffix('.receipt.json').write_text(json.dumps({'schema':'bhl-native-feature-diagnostic-plot-v1',
        'source_files':sources,'script_sha256':sha256(__file__),'plot_sha256':sha256(args.output),
        'scope':'Actual native telemetry from original consumed capture, no new confirmation claim'},indent=2)+'\n')


if __name__=='__main__':main()
