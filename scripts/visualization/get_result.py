from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np

from utils.common import load_json


def get_dev_losses(output_dir: Path):
    ckpt_dirs = output_dir.glob('ckpt-*')
    dev_scores = {
        'mse': [],
        'nmse': [],
    }
    for ckpt_dir in ckpt_dirs:
        scores = load_json(ckpt_dir / 'dev_scores.json')['mean']
        for key in dev_scores:
            dev_scores[key].append(scores[key])
    return dev_scores


def get_test_scores(output_dir: Path):
    test_dir = output_dir / 'test'
    scores = load_json(test_dir / 'scores.json')
    # print(list(scores.keys()))
    if 'scores' in scores:
        scores = scores['scores']
    if 'mean' in scores:
        return scores['mean']
    # print(scores['mean'])
    avgs = {}
    for key in scores:
        avgs[key] = np.mean(scores[key])
    return avgs


def get_data_result(data_param_dir: Path, model_pattern: str = "*") -> list[list]:
    print(f'getting result for {data_param_dir}')
    scores = []
    # run_dirs = [run_dir for run_dir in run_dirs if 'depthb4' not in run_dir.name]
    for model_dir in sorted(data_param_dir.glob(model_pattern)):
        run_dirs = sorted(model_dir.glob('*'))
        run_dirs = [run_dir for run_dir in run_dirs]
        for run_dir in run_dirs:
            '''
            Each run_dir correspond to one set of hyperparameters. E.g.,
            dam_bc_geo/dt0.1/fno/lr_0.0001_d4_h32_m112_m212
            '''
            if not run_dir.is_dir():
                continue
            try:
                test_scores = get_test_scores(run_dir)
                scores.append([str(run_dir)] + list(test_scores.values()))
            except FileNotFoundError:
                print(run_dir.name, 'not found')
                pass
    return scores


def get_result(result_dir: Path, data_pattern: str, model_pattern: str):
    data_dirs = result_dir.glob(data_pattern)
    scores = []
    for data_dir in data_dirs:
        if not data_dir.is_dir():
            continue
        # if 'prop' in data_dir.name:
        #     continue
        # Loop data subdirs such as `dt0.1`
        for data_param_dir in data_dir.iterdir():
            if data_param_dir.is_dir():
                scores += get_data_result(data_param_dir, model_pattern=model_pattern)
    table = [[str(score) for score in line] for line in scores]
    # table = sorted(table)
    # transpose
    rows = []
    n_cols = len(table[0])
    for c in range(n_cols):
        row = [line[c] for line in table]
        rows.append(row)

    lines = ['\t'.join(line) for line in rows]

    print(*lines)
    data_pattern = data_pattern.replace('*', '+')
    out_path = result_dir / f'{data_pattern}_{model_pattern}.txt'
    print(*lines, sep='\n', file=open(out_path, 'w', encoding='utf8'))


if __name__ == '__main__':
    result_dir = Path('result/auto')
    data_pattern = 'dam*'
    model_pattern = "auto_edeeponet"
    # model_pattern = "auto_ffn"
    model_pattern = "auto_deeponet_cnn"
    model_pattern = "fno"
    # model_pattern = '*'
    # model_pattern = "ffn"
    # model_pattern = "deeponet"

    get_result(result_dir, data_pattern, model_pattern)
