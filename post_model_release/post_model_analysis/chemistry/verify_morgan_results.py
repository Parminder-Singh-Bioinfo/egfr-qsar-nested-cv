"""Verify saved bits, environments and summary tables; does not fit models."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog('rdApp.warning')
p = argparse.ArgumentParser()
p.add_argument('--component-root', required=True)
p.add_argument('--output', required=True)
args = p.parse_args()
root = Path(args.component_root)
ref = root / 'reference_results'
data = root / 'analysis_inputs'
output = Path(args.output)
if output.exists():
    raise FileExistsError(output)
checks = []

def check(name, passed):
    checks.append({'check': name, 'passed': bool(passed)})
    if not passed:
        raise AssertionError(name)

labels = pd.read_csv(data / 'Error_Analysis/external_predictions_with_error_groups.csv')
smiles = pd.read_csv(ref / 'input/single_protein_external_smiles_2524.csv')
matrix = pd.read_csv(data / 'Main_SHAP/shap_molecular_inputs.csv')
config = json.loads((ref / 'run_configuration.json').read_text())
joined = labels.merge(smiles[['external_row_index', 'SMILES_STD']], on='external_row_index', validate='one_to_one', how='left')
check('Complete ordered external row mapping', len(joined) == len(matrix) == 2524 and joined.SMILES_STD.notna().all() and np.array_equal(joined.external_row_index, np.arange(len(matrix))))
columns = [c for c in matrix if c.startswith('rdkit_morgan_r2_2048bit_')]
check('Exactly 300 binary molecular columns', len(columns) == 300 and matrix[columns].isin([0, 1]).all().all())
bits = [int(c.rsplit('_', 1)[1]) for c in columns]
saved_bits = matrix[columns].to_numpy()
records = []
mismatch = 0
for i, row in joined.iterrows():
    mol = Chem.MolFromSmiles(row.SMILES_STD)
    if mol is None:
        raise ValueError('Invalid structure at row ' + str(i))
    info = {}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False, useFeatures=False, useBondTypes=True, bitInfo=info)
    mismatch += int(np.count_nonzero(np.array([fp.GetBit(bit) for bit in bits]) != saved_bits[i]))
    for bit in config['selected_bits']:
        for center, radius in info.get(bit, []):
            bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center)) if radius else []
            atoms = {center}
            for b in bonds:
                bond = mol.GetBondWithIdx(b)
                atoms.update([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            kw = dict(atomsToUse=sorted(atoms), canonical=True, isomericSmiles=True)
            if radius:
                kw['bondsToUse'] = bonds
            fragment = Chem.MolFragmentToSmiles(mol, **kw)
            records.append(dict(external_row_index=i, bit=bit, center_atom=center, radius=radius, fragment_smiles=fragment, environment_key='r%d:%s' % (radius, fragment)))
check('All 757200 saved model-bit values reconstructed', mismatch == 0)
env = pd.DataFrame(records)
saved = pd.read_csv(ref / 'tables/morgan_atom_environments.csv')
cols = list(env.columns)
pd.testing.assert_frame_equal(env[cols].reset_index(drop=True), saved[cols].reset_index(drop=True), check_dtype=False)
check('Every atom environment matches saved records', True)
summary = env.groupby(['bit','environment_key','fragment_smiles','radius']).agg(environment_occurrences=('external_row_index','size'), unique_observations=('external_row_index','nunique')).reset_index()
pd.testing.assert_frame_equal(summary, pd.read_csv(ref / 'tables/environment_frequency.csv'), check_dtype=False)
check('Environment occurrence and observation frequencies', True)
div = summary.groupby('bit').agg(distinct_environment_representations=('environment_key','nunique'), total_environment_occurrences=('environment_occurrences','sum')).reset_index()
div['multiple_environments_detected'] = div.distinct_environment_representations > 1
pd.testing.assert_frame_equal(div, pd.read_csv(ref / 'tables/bit_environment_diversity.csv'), check_dtype=False)
check('Bit environment diversity summaries', True)
prev = pd.read_csv(ref / 'tables/tp_fn_bit_prevalence.csv')
for row in prev.itertuples():
    mask = joined.group.eq(row.group)
    values = matrix.loc[mask, 'rdkit_morgan_r2_2048bit_%04d' % row.bit]
    check('Bit %d %s prevalence' % (row.bit,row.group), len(values) == row.n and int(values.sum()) == row.bit_present_n and np.isclose(100*values.mean(), row.prevalence_percent, atol=1e-10, rtol=0))
rep = pd.read_csv(ref / 'tables/representative_environment_images.csv')
for row in rep.itertuples():
    match = env[(env.bit == row.bit) & (env.external_row_index == row.external_row_index) & (env.radius == row.radius) & (env.fragment_smiles == row.fragment_smiles)]
    check('Representative %d/%d membership and image' % (row.bit,row.environment_rank), len(match) > 0 and joined.iloc[row.external_row_index]['group'] == row.group and (ref / 'figures/individual_bits' / row.image_file).is_file())
report = {'all_passed': True, 'checks': checks, 'rdkit_version': rdBase.rdkitVersion, 'reconstructed_bit_values':len(matrix)*len(columns), 'mismatches': mismatch, 'environment_records':len(env), 'model_training_performed':False, 'shap_recalculated':False}
output.write_text(json.dumps(report, indent=2) + '\n')
print('Passed %d checks; %d bit values, %d environments' % (len(checks), len(matrix)*len(columns), len(env)))
