import argparse
import pandas as pd
import numpy as np
import pickle as pl
import os
import json

from sklearn.metrics import mean_squared_error
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats

import sys
sys.path.insert(0, os.path.dirname(__file__))
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from correlation_stats import dependent_corr


def evaluate_experiment(experiment, model_dir, folds=5):
    """Evaluate one experiment and return gene-level tables plus summary metrics."""
    exp_dir = os.path.join(model_dir, experiment)
    pkl_path = os.path.join(exp_dir, 'test_results.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f'[{experiment}] test_results.pkl not found: {pkl_path}')

    print(f'[{experiment}] Evaluating...')
    with open(pkl_path, 'rb') as f:
        test_res = pl.load(f)
    meta_path = os.path.join(exp_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f'[{experiment}] metadata.json not found: {meta_path}')
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    real_list, pred_list, random_list = [], [], []
    genes = test_res['genes']

    for k in range(folds):
        data = test_res[f'split_{k}']
        if 'random' not in data:
            raise KeyError(
                f'[{experiment}] split_{k} lacks the untrained random prediction required by the recovered-gene rule.'
            )
        real_list.append(pd.DataFrame(data['real'], index=data['wsi_file_name'], columns=genes))
        pred_list.append(pd.DataFrame(data['preds'], index=data['wsi_file_name'], columns=genes))
        random_list.append(pd.DataFrame(data['random'], index=data['wsi_file_name'], columns=genes))

    df_real = pd.concat(real_list)
    df_pred = pd.concat(pred_list)
    df_random = pd.concat(random_list)

    assert np.all(df_real.index == df_pred.index)
    assert np.all(df_real.index == df_random.index)

    pred_r, random_r, test_p, pearson_p = [], [], [], []
    rmse_pred, rmse_random, rmse_quantile_norm, rmse_mean_norm = [], [], [], []
    valid_genes = []

    for gene in genes:
        r = df_real.loc[:, gene]
        p = df_pred.loc[:, gene]
        rand = df_random.loc[:, gene]

        if len(set(p)) == 1 or len(set(r)) == 1 or len(set(rand)) == 1:
            xy, xz, yz = 0, 0, 0
            p1, p2, p3, pv = 1, 1, 1, 1
        else:
            xy, p1 = stats.pearsonr(r, p)
            xz, p2 = stats.pearsonr(r, rand)
            yz, p3 = stats.pearsonr(p, rand)
            _, pv = dependent_corr(xy, xz, yz, len(r), twotailed=False, conf_level=0.95, method='steiger')

        pred_r.append(xy)
        random_r.append(xz)
        test_p.append(pv)
        pearson_p.append(p1)

        rmse_p = np.sqrt(mean_squared_error(r, p))
        rmse_r = np.sqrt(mean_squared_error(r, rand))
        rmse_q = rmse_p / (np.quantile(r, 0.75) - np.quantile(r, 0.25) + 1e-5)
        rmse_m = rmse_p / (np.mean(r) + 1e-8)

        rmse_pred.append(rmse_p)
        rmse_random.append(rmse_r)
        rmse_quantile_norm.append(rmse_q)
        rmse_mean_norm.append(rmse_m)
        valid_genes.append(gene)

    combine_res = pd.DataFrame({
        'pred_real_r': pred_r,
        'random_real_r': random_r,
        'pearson_p': pearson_p,
        'Steiger_p': test_p,
        'rmse_pred': rmse_pred,
        'rmse_random': rmse_random,
        'rmse_quantile_norm': rmse_quantile_norm,
        'rmse_mean_norm': rmse_mean_norm,
    }, index=valid_genes)

    combine_res = combine_res.sort_values('pred_real_r', ascending=False)
    combine_res['pred_real_r'] = combine_res['pred_real_r'].fillna(0)
    combine_res['random_real_r'] = combine_res['random_real_r'].fillna(0)

    combine_res['pearson_p'] = combine_res['pearson_p'].fillna(1)
    _, fdr_pearson_p = fdrcorrection(combine_res['pearson_p'])
    combine_res['fdr_pearson_p'] = fdr_pearson_p

    combine_res['Steiger_p'] = combine_res['Steiger_p'].fillna(1)
    _, fdr_Steiger_p = fdrcorrection(combine_res['Steiger_p'])
    combine_res['fdr_Steiger_p'] = fdr_Steiger_p

    cancer = metadata.get('cancer') or os.path.basename(os.path.dirname(os.path.dirname(model_dir)))
    combine_res['cancer'] = cancer

    # Normalize RMSE by the observed IQR, then min-max scale within the experiment.
    col = combine_res['rmse_quantile_norm']
    mn, mx = col.min(), col.max()
    combine_res['nrmse_01'] = (col - mn) / (mx - mn + 1e-8)

    # Recovered genes under the paired dependent-correlation criterion.
    sig_res = combine_res[
        (combine_res['pred_real_r'] > 0) &
        (combine_res['pearson_p'] < 0.05) &
        (combine_res['rmse_pred'] < combine_res['rmse_random']) &
        (combine_res['pred_real_r'] > combine_res['random_real_r']) &
        (combine_res['Steiger_p'] < 0.05) &
        (combine_res['fdr_Steiger_p'] < 0.2)
    ]

    num_sig = len(sig_res)
    top1000 = combine_res.head(1000)  # combine_res is sorted by PCC descending.
    parameter_count_excluding_output_head = metadata.get('parameter_count_excluding_output_head')
    parameter_count_including_output_head = metadata.get('parameter_count_including_output_head')
    if parameter_count_excluding_output_head is None or parameter_count_including_output_head is None:
        raise ValueError(
            f'[{experiment}] metadata.json must contain both '
            f'parameter_count_excluding_output_head and parameter_count_including_output_head.'
        )

    head_type = metadata.get('head_type')
    if head_type == 'morsa':
        head_type = 'spex'
    basis_type = metadata.get('basis_type')
    if basis_type is None:
        basis_type = {
            'spex': 'pca',
            'learned_rank': 'learned',
            'covnull_spex': 'covariance_null_pca',
        }.get(head_type)
    model_type = metadata.get('model_type')
    encoder = metadata.get('encoder')
    if encoder is None:
        encoder = {
            'mean': 'mean',
            'he2rna': 'he2rna',
            'vis': 'vis',
            'morsa_enc': 'spd',
            'diag_morsa_enc': 'diag_spd',
            'diag_spd': 'diag_spd',
        }.get(model_type, model_type)

    summary = {
        'experiment': experiment,
        'cancer': cancer,
        'num_sig_genes': num_sig,
        'top1000_mean_pcc': top1000['pred_real_r'].mean(),
        'all_mean_pcc': combine_res['pred_real_r'].mean(),
        'top1000_mean_rmse': top1000['rmse_pred'].mean(),
        'all_mean_rmse': combine_res['rmse_pred'].mean(),
        'top1000_mean_nrmse': top1000['nrmse_01'].mean(),
        'all_mean_nrmse': combine_res['nrmse_01'].mean(),
        'avg_fold_train_seconds': metadata.get('avg_fold_train_seconds'),
        'sum_fold_train_seconds': metadata.get('sum_fold_train_seconds'),
        'parameter_count': parameter_count_excluding_output_head,
        'parameter_count_excluding_output_head': parameter_count_excluding_output_head,
        'parameter_count_including_output_head': parameter_count_including_output_head,
        'peak_gpu_memory_mb': metadata.get('peak_gpu_memory_mb'),
        'avg_inference_time_per_wsi_seconds': metadata.get('avg_inference_time_per_wsi_seconds'),
        'model_type': model_type,
        'encoder': encoder,
        'head_type': head_type,
        'basis_type': basis_type,
        'rank_k': metadata.get('rank_k', metadata.get('morsa_k')),
        'training_seed': metadata.get('training_seed', 29),
        'split_seed': metadata.get('split_seed', 0),
        'basis_seed': metadata.get('basis_seed'),
        'config_hash': metadata.get('config_hash'),
    }

    print(f'[{experiment}] sig_genes={num_sig}, all_pcc={summary["all_mean_pcc"]:.4f}, '
          f'top1000_pcc={summary["top1000_mean_pcc"]:.4f}')

    return combine_res, sig_res, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate experiment results')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name (subdirectory under model_dir)')
    parser.add_argument('--model_dir', type=str, required=True, help='TCGA output directory')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    args = parser.parse_args()

    combine_res, sig_res, summary = evaluate_experiment(
        args.experiment, args.model_dir, args.folds
    )

    save_path = os.path.join(args.model_dir, 'results', args.experiment)
    os.makedirs(save_path, exist_ok=True)

    combine_res.to_csv(os.path.join(save_path, 'all_genes.csv'))
    sig_res.to_csv(os.path.join(save_path, 'sig_genes.csv'))

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(save_path, 'summary_row.csv'), index=False)

    print(f'[{args.experiment}] Results saved to {save_path}')
