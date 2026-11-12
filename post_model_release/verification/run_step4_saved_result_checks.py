#!/usr/bin/env python3
"""Step 4: staged saved-result analysis and figure replay; no model fitting."""
import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--staging', type=Path, default=Path.home() / 'projects/EGFR_REPOSITORY_ADDITIONS_STAGING')
    p.add_argument('--repository-root', type=Path, default=Path.home() / 'projects/EGFR_QSAR_PUBLICATION_RELEASE')
    p.add_argument('--check-only', action='store_true', help='Verify staged checksums and syntax without running analyses')
    args = p.parse_args()
    root = args.staging.resolve()
    repo = args.repository_root.resolve()
    data = root / 'data'
    if not (root / 'STEP3_REPORT.json').is_file():
        raise SystemExit('Step 3 report not found. Finish Step 3 first.')
    for manifest in ['STEP3_INPUT_CHECKSUMS.csv', 'STEP3_CODE_CHECKSUMS.csv']:
        with (root / manifest).open(newline='') as f:
            entries = list(csv.DictReader(f))
        for entry in entries:
            path = root / entry['relative_path']
            if sha(path) != entry['sha256']:
                raise ValueError('Staged file differs from Step 3 checksum: ' + str(path))
    for folder in ['post_model_analysis', 'visualization']:
        for path in (root / folder).rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
    # Scientific modules used by the saved-result checks, not estimator fitting.
    import numpy as np
    import pandas as pd
    import matplotlib
    import sklearn
    from rdkit import rdBase

    sp = repo / 'single_protein/results/final_nested_qsar/full_cutoff_7'
    cb = repo / 'cell_based/results/final_nested_qsar/literature_multisource_token_supported/full_cutoff_7'
    for base in [sp, cb]:
        for rep in ['2d', '3d', 'morgan', '2d_3d', '2d_3d_morgan']:
            path = base / rep / 'predictions/strict_external_predictions.csv'
            if not path.is_file():
                raise FileNotFoundError(path)
    if args.check_only:
        print('Step 4 preflight passed: checksums, Python syntax, imports and saved prediction paths.')
        print('No analyses executed. No outputs created.')
        return

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = root / 'verification_runs' / ('step4_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    logs = out / 'logs'
    logs.mkdir()
    checks = []
    report = {'all_passed':False, 'model_training_performed':False, 'shap_recalculated':False,
              'output_directory':str(out), 'checks':checks,
              'scope':'Saved-result runtime checks and plot rendering; no nested-CV retraining, ablation refitting or new SHAP explanations.',
              'versions':{'python':sys.version.split()[0], 'numpy':np.__version__,
                          'pandas':pd.__version__, 'matplotlib':matplotlib.__version__,
                          'scikit_learn':sklearn.__version__, 'rdkit':rdBase.rdkitVersion},
              'visual_qa':'Rendered outputs still require visual inspection; numerical checks do not establish pixel identity.'}
    def save():
        (out / 'STEP4_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    def check(name, passed):
        checks.append({'check':name, 'passed':bool(passed)})
        save()
        if not passed:
            raise AssertionError(name)
    def run(label, script, *options):
        print('RUNNING:', label, flush=True)
        env = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
        log = logs / (label + '.log')
        with log.open('w') as f:
            result = subprocess.run([sys.executable,str(root/script)] + [str(x) for x in options],
                                    stdout=f,stderr=subprocess.STDOUT,cwd=str(root),env=env)
        if result.returncode:
            print(log.read_text(errors='replace')[-5000:], flush=True)
            raise RuntimeError(label + ' failed; see ' + str(log))
        check(label + '/command completed',True)
    def compare(name, new, reference):
        a, b = pd.read_csv(new), pd.read_csv(reference)
        pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,atol=1e-10,rtol=0)
        check(name,True)
    def image_set(folder, stem):
        for ext in ['pdf','svg','png','tiff']:
            path = folder / (stem + '.' + ext)
            check(str(path.relative_to(out)) + '/nonempty',path.is_file() and path.stat().st_size>0)

    save()
    try:
        # 1. All four corrected model-performance figures.
        modelplots = out / 'figures/model_performance'
        run('01_model_figures','visualization/model_performance/render_all.py',
            '--input-dir',data/'figure_inputs/model_performance','--output-dir',modelplots)
        for n,stem in [(2,'Fig2_inner_cv'),(3,'Fig3_outer_cv'),(4,'Fig4_development_cv'),(5,'Fig5_external_holdout')]:
            image_set(modelplots / ('Fig%d'%n),stem)
        for n,stem in [(3,'Fig3_outer_cv'),(5,'Fig5_external_holdout')]:
            svg = (modelplots/('Fig%d'%n)/(stem+'.svg')).read_text()
            caption = (modelplots/('Fig%d'%n)/('caption_Fig%d.txt'%n)).read_text()
            check('Figure %d/AP axis and caption'%n,'PR-AUC' not in svg and 'AP' in svg and 'average precision (AP)' in caption)
        raw = pd.read_csv(data/'figure_inputs/model_performance/combined_single_protein_cell_based_outer_cv.csv')
        summary = pd.read_csv(modelplots/'Fig3/outer_cv_summary.csv')
        for item in summary.itertuples():
            values = raw.loc[raw.Dataset.eq(item.Dataset)&raw.Representation.eq(item.Representation),item.Metric]
            check('Outer CV summary/'+item.Dataset+'/'+item.Representation+'/'+item.Metric,
                  len(values)==5 and np.allclose([values.mean(),values.std(ddof=1)],[item.Mean,item.SD],atol=1e-10,rtol=0))
        raw_ad = pd.read_csv(data/'figure_inputs/model_performance/combined_applicability_domain_results.csv')
        plotted_ad = pd.read_csv(modelplots/'Fig5/external_validation_summary.csv')
        cols = ['budget','rows','ad_coverage_percent'] + [m+s for m in ['mcc','pr_auc','roc_auc'] for s in ['','_ci_low','_ci_high']]
        pd.testing.assert_frame_equal(raw_ad[cols],plotted_ad[cols],check_dtype=False,check_exact=False,atol=1e-10,rtol=0)
        check('Figure 5/saved estimates and intervals unchanged',True)

        # 2. Scaffold metrics recalculated from all ten saved prediction tables.
        scaffold = out/'results/scaffold'
        run('02_scaffold_analysis','post_model_analysis/scaffold/analyze_scaffold_performance.py',
            '--single-scaffold-dir',data/'scaffold/single_protein','--single-predictions-root',sp,
            '--cell-scaffold-dir',data/'scaffold/cell_based','--cell-predictions-root',cb,'--output-dir',scaffold)
        for name in ['all_10_scaffold_performance.csv','all_10_scaffold_ad_performance.csv',
                     'all_10_scaffold_summary.csv','original_scaffold_reproducibility_check.csv']:
            compare('Scaffold/'+name,scaffold/name,data/'figure_inputs/scaffold'/name)
        run('03_scaffold_figure','visualization/scaffold/plot_scaffold_performance.py',
            '--performance',scaffold/'all_10_scaffold_performance.csv',
            '--ad-performance',scaffold/'all_10_scaffold_ad_performance.csv','--output-dir',out/'figures/scaffold')
        image_set(out/'figures/scaffold','FigC_scaffold_bars_heatmap')

        # 3. Mutation ablation: rebuild comparison from saved predictions, not refitting.
        ablation = out/'results/ablation'
        run('04_ablation_comparison','post_model_analysis/ablation/build_ablation_comparison.py',
            '--reference-dir',data/'results/ablation','--output-dir',ablation)
        a = pd.read_csv(ablation/'ablation_comparison.csv')
        check('Ablation/two models and six removed indicators',len(a)==2 and a.Mutation_features_removed.eq(6).all())
        check('Ablation/full-minus-ablated MCC arithmetic',np.allclose(a.MCC_full-a.MCC_ablated,a.MCC_full_minus_ablated,atol=1e-10,rtol=0))

        # 4. Recompute SHAP subgroup summaries from saved molecular SHAP values.
        shapplotinputs = out/'replay_inputs/shap'
        shutil.copytree(data/'figure_inputs/shap',shapplotinputs)
        groups = {'03_Classification_Errors':'Full_External',
                  '04_AD_Analysis/Inside_AD':'Inside_AD','04_AD_Analysis/Outside_AD':'Outside_AD',
                  '05_Scaffold_Analysis/Scaffold_Seen':'Scaffold_Seen','05_Scaffold_Analysis/Scaffold_Unseen':'Scaffold_Unseen'}
        for assay,folder,script in [
            ('single_protein','01_Single_Protein_Morgan','analyze_single_protein_integrated_shap.py'),
            ('cell_based','02_Cell_Based_3D','analyze_cell_based_integrated_shap.py')]:
            target = out/'results/shap'/assay
            run('05_shap_'+assay,'post_model_analysis/shap/'+script,
                '--repository-root',repo,'--data-root',data,'--output-dir',target)
            for staged_group,generated_group in groups.items():
                for name in ['TP_vs_FN_feature_comparison.csv','TN_vs_FP_feature_comparison.csv']:
                    reference=data/'figure_inputs/shap'/folder/staged_group/name
                    current=target/generated_group/name
                    check('SHAP/'+assay+'/'+generated_group+'/'+name+'/expected presence',reference.exists()==current.exists())
                    if reference.exists():
                        compare('SHAP/'+assay+'/'+generated_group+'/'+name,current,reference)
                        shutil.copy2(current,shapplotinputs/folder/staged_group/name)
        run('06_shap_figures','visualization/shap/plot_shap_figures.py',
            '--input-root',shapplotinputs,'--output-dir',out/'figures/shap')
        for stem in ['Fig_Single_protein_Morgan','Fig_Cell_based_3D']:
            image_set(out/'figures/shap',stem)

        # 5. Reconstruct all 300 Morgan bits and selected environments, then replay plot.
        run('07_morgan_verification','post_model_analysis/chemistry/verify_morgan_results.py',
            '--component-root',data/'chemistry','--output',out/'results/morgan_verification.json')
        mr=json.loads((out/'results/morgan_verification.json').read_text())
        check('Morgan/all independent checks passed',mr['all_passed'] and mr['mismatches']==0)
        run('08_morgan_figure','visualization/chemistry/plot_morgan_chemistry.py',
            '--input-root',data/'chemistry/reference_results','--output-dir',out/'figures/morgan')
        image_set(out/'figures/morgan','Fig_Morgan_chemical_interpretation')

        # Sources/inputs must still match their pre-run checksums.
        for manifest in ['STEP3_INPUT_CHECKSUMS.csv','STEP3_CODE_CHECKSUMS.csv']:
            with (root/manifest).open(newline='') as f:
                for entry in csv.DictReader(f):
                    if sha(root/entry['relative_path']) != entry['sha256']:
                        raise ValueError('Staged input/source changed during replay: '+entry['relative_path'])
        check('Staged inputs and sources unchanged after replay',True)
        report['all_passed']=True
        save()
        with (out/'OUTPUT_CHECKSUMS.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['relative_path','bytes','sha256'])
            for path in sorted(out.rglob('*')):
                if path.is_file() and path.name!='OUTPUT_CHECKSUMS.csv':
                    writer.writerow([path.relative_to(out).as_posix(),path.stat().st_size,sha(path)])
        print('\nSTEP 4 PASSED:',len(checks),'checks',flush=True)
        print('Outputs and logs:',out,flush=True)
        print('No model training or new SHAP explanations. Nothing uploaded.',flush=True)
        print('Next: visually inspect regenerated figures, then prepare upload archives.',flush=True)
    except Exception as error:
        report['error']=str(error)
        save()
        print('\nSTEP 4 STOPPED. Report:',out/'STEP4_REPORT.json',flush=True)
        raise


if __name__=='__main__':
    main()
