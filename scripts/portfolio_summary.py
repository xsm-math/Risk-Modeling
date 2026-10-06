"""Presentation-only charts from committed aggregate outputs; no model execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, SLATE, TEAL, ORANGE = '#236d91', '#8795a5', '#258a75', '#b26b32'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', choices=['inventory', 'credit', 'gold'], required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    sources = []
    def rows(name):
        path = root / name
        sources.append(path)
        with path.open(encoding='utf-8', newline='') as handle:
            return list(csv.DictReader(handle))
    def panel(ax, labels, values, title, unit, colors, error=None):
        ax.barh(labels, values, color=colors, xerr=error, capsize=3, height=.58)
        ax.invert_yaxis()
        ax.set_title(title, loc='left', fontweight='bold', pad=12)
        ax.set_xlabel(unit)
        ax.axvline(0, color=SLATE, linewidth=.7)
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
        ax.margins(x=.22)
        for i, value in enumerate(values):
            end=value+(error[i] if value>=0 else -error[i]) if error is not None else value
            ax.annotate(f'{value:.2f}', (end, i), xytext=(5 if value >= 0 else -5, 0), textcoords='offset points', va='center', ha='left' if value >= 0 else 'right', fontsize=9)
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'axes.spines.left':False, 'svg.fonttype':'none', 'svg.hashsalt':'quantitative-decision-science'}):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        if args.project == 'inventory':
            pairs = [r for r in rows('results/network/paired_comparisons.csv') if r['policy']=='mpc_buffered']
            agg = {(r['regime'],r['policy']):r for r in rows('results/network/aggregate.csv')}
            labels = [r['regime'].replace('_',' ').title() for r in pairs]
            panel(axes[0], labels, [float(r['mean_paired_gain'])/1000 for r in pairs], 'Buffered MILP: paired economic gain', 'CNY thousands vs. base stock', [BLUE]*3, [float(r['ci95_halfwidth'])/1000 for r in pairs])
            panel(axes[1], labels, [100*(float(agg[r['regime'],'mpc_buffered']['fill_rate'])-float(agg[r['regime'],'base_stock']['fill_rate'])) for r in pairs], 'Aggregate service trade-off', 'Fill-rate difference / percentage points', [ORANGE]*3)
            note='Synthetic network; 12 paired seeds per regime. Left error bars: unadjusted 95% t intervals.'
        elif args.project == 'credit':
            selected=json.loads((root/'reports/credit/selection.json').read_text())['model']
            sources.append(root/'reports/credit/selection.json')
            data={r['model']:r for r in rows('reports/credit/test_metrics.csv')}
            names=['woe_lr',selected]; labels=['WOE scorecard','Selected calibrated RF']
            panel(axes[0],labels,[float(data[n]['auc']) for n in names], 'Frozen test ranking', 'ROC-AUC (higher is better)',[SLATE,BLUE])
            axes[0].set_xlim(0,1)
            for text, n in zip(axes[0].texts,names): text.set_text(f"{float(data[n]['auc']):.4f}")
            panel(axes[1],labels,[float(data[n]['brier']) for n in names], 'Frozen test probability error', 'Brier score (lower is better)',[SLATE,BLUE])
            for text, n in zip(axes[1].texts,names): text.set_text(f"{float(data[n]['brier']):.4f}")
            note='UCI historical cohort; model selected by validation log loss. Reused split; no external holdout.'
        else:
            data={r['Strategy']:r for r in rows('reports/reference/metrics.csv') if r['Period']=='2020-2025'}
            names=['buy_hold','fixed_50','trend','vol']
            labels=['Buy and hold','Fixed 50%','Trend timing','Volatility targeting']
            panel(axes[0],labels,[100*float(data[n]['CAGR']) for n in names], 'Historical annualized return', 'CAGR / %', [SLATE,TEAL,ORANGE,BLUE])
            panel(axes[1],labels,[-100*float(data[n]['MaxDrawdown']) for n in names], 'Historical maximum drawdown', 'Loss magnitude / %', [SLATE,TEAL,ORANGE,BLUE])
            note='GLD, 2020–2025; 10 bps per side, zero cash return. Exposure differs; retrospective comparison.'
        fig.subplots_adjust(left=.18,right=.97,top=.87,bottom=.25,wspace=.8)
        fig.text(.02,.04,note,fontsize=9,color='#475569')
        out=root/'assets/portfolio';out.mkdir(parents=True,exist_ok=True)
        fig.savefig(out/'summary.svg',metadata={'Date':None})
        fig.savefig(out/'summary.png',dpi=150)
        plt.close(fig)
        manifest={'project':args.project,'chart':'summary.svg','scope':'Presentation from existing aggregate outputs; no refitting or new experiment.', 'inputs':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sources},'matplotlib':matplotlib.__version__}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        print('Generated',args.project,'summary from',len(sources),'committed evidence files.')

if __name__=='__main__':
    main()
