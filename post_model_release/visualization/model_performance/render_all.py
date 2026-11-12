"""Render four manuscript plots from saved results, without training."""
import argparse
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--input-dir',required=True,type=Path)
p.add_argument('--output-dir',required=True,type=Path)
args = p.parse_args()
here = Path(__file__).resolve().parent
jobs = [
    ('plot_fig2_inner_cv.py','all_350_inner_cv_results.csv','Fig2'),
    ('plot_fig3_outer_cv.py','combined_single_protein_cell_based_outer_cv.csv','Fig3'),
    ('plot_fig4_development_cv.py','combined_development_search_plot_data.csv','Fig4'),
    ('plot_fig5_external_holdout.py','combined_applicability_domain_results.csv','Fig5'),
]
if args.output_dir.exists() and any(args.output_dir.iterdir()):
    raise FileExistsError('Output root must be new or empty')
for script,table,folder in jobs:
    if not (args.input_dir/table).is_file():
        raise FileNotFoundError(args.input_dir/table)
for script,table,folder in jobs:
    cmd=[sys.executable,str(here/script),'--input',str(args.input_dir/table),'--output-dir',str(args.output_dir/folder)]
    subprocess.run(cmd,check=True)
print('Four figure scripts completed; no training performed.')
