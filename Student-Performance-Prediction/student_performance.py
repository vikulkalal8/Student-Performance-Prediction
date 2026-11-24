"""Run a reproducible experiment and export results for the report."""
import argparse, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from model import ROOT, available, experiment, predict

def export(mode):
    r,_ = experiment(mode); out=ROOT/'output'/mode; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(r['comparison']).to_csv(out/'model_comparison.csv',index=False)
    (out/'experiment.json').write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf-8')
    k=r['selected']; actual=r['actual']; pred=r['predictions'][k]; baseline=r['predictions']['baseline']
    fig,ax=plt.subplots(figsize=(6.6,4.0)); lo=min(0,*pred,*baseline); hi=max(r['scale'],*pred,*baseline)
    ax.plot([lo,hi],[lo,hi],ls='--',color='#8896a6',label='Perfect prediction'); ax.scatter(actual,baseline,facecolors='none',edgecolors='#365c90',label='No PCA'); ax.scatter(actual,pred,color='#087c76',label=f'{k} PCs'); ax.set(xlabel='Actual final marks',ylabel='Predicted final marks',title='Held-out predictions'); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'actual_vs_predicted.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.6,3.7)); ax.bar(range(1,len(r['explained'])+1),r['explained'],color='#087c76'); ax.set(xlabel='Principal component',ylabel='Explained variance (%)',title='Training-set PCA variance',xticks=range(1,len(r['explained'])+1)); ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(out/'explained_variance.png',dpi=160); plt.close(fig)
    values={f:b['default'] if mode=='synthetic' else round(b['default']) for f,b in r['bounds'].items()}
    print(f'\n{mode}: {r["rows"]} records, selected {k} PCs by training CV'); print(pd.DataFrame(r['comparison']).round(4).to_string(index=False)); print('Example prediction:',round(predict(mode,k,values)['prediction'],3)); print('Saved:',out)
    return r
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['all','synthetic','uci','uci_early'],default='all'); a=p.parse_args()
    modes=available() if a.dataset=='all' else [a.dataset]
    for mode in modes:
        if mode not in available(): p.exit(1,'Dataset missing. Run python fetch_dataset.py first.\n')
        export(mode)
