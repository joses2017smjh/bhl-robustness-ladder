#!/usr/bin/env python3
"""Recollect downloaded raw fusion evidence and draw measured tradeoff figures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import fusion_occlusion_campaign as CAMPAIGN
from h34_campaign import safe_extract

LABELS={'lidar_sparse':'L','sgbm':'S','foundation':'F','sgbm_simple':'SS','sgbm_agreement':'SA',
        'sgbm_cost':'SC','foundation_simple':'FS','foundation_agreement':'FA','foundation_cost':'FC'}
FULL_LABELS={'lidar_sparse':'Sparse LiDAR','sgbm':'SGBM','foundation':'FoundationStereo',
             'sgbm_simple':'SGBM + simple fusion','sgbm_agreement':'SGBM + agreement gate',
             'sgbm_cost':'SGBM + local-cost gate','foundation_simple':'Foundation + simple fusion',
             'foundation_agreement':'Foundation + agreement gate','foundation_cost':'Foundation + local-cost gate'}
MARKERS=['o','s','^','D','v','P','X','<','>']


def pooled(summary):
    rows=[]
    for condition in ('clear','partial','heavy'):
        for arm in CAMPAIGN.ARMS:
            selected=[r for r in summary['group_metrics'] if r['condition']==condition and r['arm']==arm]
            eligible=sum(r['eligible_pixels'] for r in selected);predicted=sum(r['predicted_pixels'] for r in selected)
            near=sum(r['near_obstacle_pixels'] for r in selected)
            false=sum(r['false_free_obstacle_pixels'] for r in selected);missed=sum(r['missed_obstacle_pixels'] for r in selected)
            rows.append({'condition':condition,'arm':arm,'geometry_groups':len(selected),'independent_seed_clusters':summary['independent_seed_clusters'],
                'eligible_pixels':eligible,'predicted_pixels':predicted,'near_obstacle_pixels':near,
                'coverage':predicted/eligible if eligible else None,'false_free_obstacle_pixels':false,
                'false_free_rate':false/near if near else None,'missed_obstacle_pixels':missed,
                'obstacle_pixel_recall':1-missed/near if near else None,
                'unknown_obstacle_pixels':sum(r['unknown_obstacle_pixels'] for r in selected),
                'native_mae_m':sum(r['mae_m']*r['predicted_pixels'] for r in selected if r['mae_m'] is not None)/predicted if predicted else None,
                'median_cell_p95_ms':float(np.median([r['p95_pipeline_ms'] for r in selected]))})
    return rows


def figures(raw, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, ListedColormap
    from matplotlib.patches import Patch
    from PIL import Image
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'savefig.facecolor':'white','svg.fonttype':'none'})
    data=pooled(summary);CAMPAIGN.write(output/'heldout-plot-data.json',{'source':'independently recollected held-out predictions',
        'aggregation':'pixel-pooled within condition; four geometry groups, two shared-seed clusters; descriptive only','rows':data})
    maes=[r['native_mae_m'] for r in data if r['native_mae_m'] is not None]
    norm=Normalize(vmin=0,vmax=max(maes));cmap=plt.get_cmap('viridis_r')
    fig,axes=plt.subplots(1,3,figsize=(15,5.6),sharey=True)
    offsets={'L':(6,5),'S':(-16,7),'F':(-15,8),'SS':(5,12),'SA':(5,-14),'SC':(5,6),'FS':(5,14),'FA':(5,-16),'FC':(5,5)}
    for ax,condition in zip(axes,('clear','partial','heavy')):
        for marker,arm in zip(MARKERS,CAMPAIGN.ARMS):
            row=next(r for r in data if r['condition']==condition and r['arm']==arm)
            x,y=100*row['coverage'],100*row['false_free_rate'];label=LABELS[arm]
            ax.scatter(x,y,c=[row['native_mae_m']],cmap=cmap,norm=norm,marker=marker,s=85,edgecolors='black',linewidths=.6,zorder=3)
            ax.annotate(label,(x,y),xytext=offsets[label],textcoords='offset points',fontsize=9)
        ax.set_title(condition.capitalize()+' foreground occlusion');ax.set_xlabel('Valid depth coverage (%)');ax.set_xlim(-3,106);ax.grid(alpha=.2)
    axes[0].set_ylabel('False-free / near-obstacle pixels (%)')
    high=max(100*r['false_free_rate'] for r in data);axes[0].set_ylim(-max(.2,high*.06),max(1,high*1.2))
    fig.subplots_adjust(left=.065,right=.91,bottom=.27,top=.84,wspace=.13)
    cax=fig.add_axes([.925,.30,.013,.48]);fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,label='Native-support MAE (m)')
    fig.suptitle('Held-out simulated stereo–LiDAR fusion: error–coverage tradeoff',fontsize=15,y=.97)
    fig.text(.065,.15,'L = sparse LiDAR; S = SGBM; F = FoundationStereo. Suffix S = simple fusion, A = agreement gate, C = local-SAD-cost gate.',fontsize=10)
    fig.text(.065,.10,'144 held-out views; 4 geometry groups / 2 independent seed clusters. No confidence intervals or hardware claim.',fontsize=10)
    fig.text(.065,.05,'Abstention can reduce false-free rate. Read coverage and obstacle recall together; sparse LiDAR is not a dense free-space map.',fontsize=10)
    for extension in ('png','svg'):fig.savefig(output/('heldout-tradeoff.'+extension),dpi=180)
    plt.close(fig)
    # Fixed example chosen by protocol position, not performance: first held-out
    # panel seed and middle index6, same pose/texture in clear and heavy scenes.
    seed=742000;index=6;example=[]
    fig,axes=plt.subplots(2,6,figsize=(18,7.2))
    titles=['Left camera','Right camera','Foundation depth','Foundation + cost gate','Independent geometry depth','Evaluator visibility']
    colors=['#e6e6e6','#e69f00','#56b4e9','#cc79a7','#222222'];visibility_cmap=ListedColormap(colors)
    depth_cmap=plt.get_cmap('turbo').copy();depth_cmap.set_bad('#222222')
    for row,condition in enumerate(('clear','heavy')):
        folder=raw/'test'/f'test-panel_background-s{seed}-{condition}'
        left=np.asarray(Image.open(folder/f'{index:04d}-left.png'));right=np.asarray(Image.open(folder/f'{index:04d}-right.png'))
        artifact=folder/f'{index:04d}-predictions-evaluation.npz'
        with np.load(artifact,allow_pickle=False) as values:
            stereo=values['foundation_depth_m'];fusion=values['foundation_cost_depth_m'];truth=values['truth_depth_m']
            visibility=values['right_occluded'].astype(np.uint8)+2*values['lidar_occluded'].astype(np.uint8)
            visibility[~values['visibility_support']]=4
            payload=[left,right,stereo,fusion,truth,visibility]
            for col,(ax,item) in enumerate(zip(axes[row],payload)):
                if col in (0,1):ax.imshow(item)
                elif col==5:ax.imshow(item,cmap=visibility_cmap,vmin=0,vmax=4,interpolation='nearest')
                else:im=ax.imshow(item,cmap=depth_cmap,vmin=.1,vmax=6.,interpolation='nearest')
                if row==0:ax.set_title(titles[col])
                ax.set_xticks([]);ax.set_yticks([])
        axes[row,0].set_ylabel(condition.capitalize(),fontsize=13)
        example.append({'condition':condition,'cell':folder.name,'frame_index':index,'artifact_sha256':CAMPAIGN.sha256_file(artifact)})
    fig.subplots_adjust(left=.04,right=.92,bottom=.22,top=.84,wspace=.05,hspace=.08)
    cax=fig.add_axes([.935,.29,.01,.47]);fig.colorbar(im,cax=cax,label='Optical depth (m; display clipped at6m)')
    fig.suptitle('Measured demonstration: one fixed held-out viewpoint, physical foreground panel added',fontsize=15,y=.96)
    legend=[Patch(facecolor=colors[i],label=t) for i,t in enumerate(['Visible in both evaluator views','Right-view occluded','LiDAR-view occluded','Both occluded','Outside/invalid evaluator support'])]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.115),ncol=5,frameon=False,fontsize=9)
    fig.text(.04,.075,'Panel layout, seed742000, frame6/12. Images and model predictions are retained simulation outputs; dark depth pixels are unavailable.',fontsize=10)
    fig.text(.04,.035,'Visibility uses independent z buffers after inference. The virtual midpoint evaluator view does not provide a third inference camera.',fontsize=10)
    for extension in ('png','svg'):fig.savefig(output/('heldout-clear-heavy-demo.'+extension),dpi=160)
    plt.close(fig)
    return data,example


def run(protocol, raw_archive, expected_sha, work, output):
    protocol,raw_archive,work,output=map(Path,(protocol,raw_archive,work,output))
    if CAMPAIGN.sha256_file(raw_archive)!=expected_sha:raise ValueError('downloaded raw archive SHA differs from publication')
    if work.exists() or output.exists():raise FileExistsError('independent extraction/report destinations must be new')
    work.mkdir();output.mkdir(parents=True)
    safe_extract(raw_archive,work)
    root_result=json.loads((work/'campaign_result.json').read_text())
    if root_result.get('status')!='PASS' or root_result.get('scientific_status')!='COMPLETE_PREDECLARED_SIMULATION_COMPARISON':raise ValueError('fullgatedcampaign did notcomplete')
    summaries={}
    for phase in ('smoke','dev','test'):
        directories=sorted(p for p in (work/phase).iterdir() if p.is_dir())
        actual=CAMPAIGN.collect(protocol,phase,directories,output/(phase+'-independent-collection.json'))
        original=json.loads((work/(phase+'-collection.json')).read_text())
        if actual!=original or actual['status']!='PASS' or not actual['readiness_gate']:raise ValueError('independent fullcollection differs: '+phase)
        summaries[phase]=actual
    data,example=figures(work,summaries['test'],output)
    receipt={'schema':'bhl-fusion-independent-report-v1','status':'PASS','raw_archive_sha256':expected_sha,
        'scientific_protocol_sha256':CAMPAIGN.sha256_file(protocol),
        'independent_recomputation':'All300frames/30cells: artifactinventory/hashes, geometricvisibility, 9armnative/common/pairwise/stratumdepth-hazardmetrics matched stored collections exactly',
        'verified_cells':{phase:s['verified_cells'] for phase,s in summaries.items()},
        'heldout_geometry_groups':summaries['test']['geometry_groups'],'heldout_independent_seed_clusters':summaries['test']['independent_seed_clusters'],
        'fixed_demo':example,'files_sha256':{p.name:CAMPAIGN.sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}}
    CAMPAIGN.write(output/'verification-receipt.json',receipt)
    print(json.dumps({'status':'PASS','verified_cells':receipt['verified_cells'],'heldout_independent_seed_clusters':receipt['heldout_independent_seed_clusters'],'output':str(output)},indent=2))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','raw-archive','work','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-sha',required=True)
    run(**vars(parser.parse_args()))
