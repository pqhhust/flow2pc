#!/usr/bin/env python3
"""Sweep HCLT students for frozen ALI-CFM/OTP-FM teachers through the shared queue.

Only the 19 teacher/dataset pairs without a previous student sweep are included,
100 configurations each. Previous sweeps count even when teachers have changed.
--dry-run is metadata-only: no Torch, GPU query, writes, W&B, or worker launch.
"""

import copy
import csv
import hashlib
import io
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path

import yaml

import run_otpfm_distill_sweep as sweep

batch = sweep.batch
ROOT = sweep.ROOT
DATASETS = batch.ALL_DATASETS
PREVIOUS_SWEEPS = {
    *(('alicfm', dataset) for dataset in
      ('cite5', 'cite50', 'cite100', 'multi5', 'multi50', 'multi100')),
    ('otpfm', 'cite50'), ('otpfm', 'cite100'),
    ('alicfm', 'cell_tracking'), ('otpfm', 'gom'), ('otpfm', 'beijing'),
}


def parser():
    cli = sweep.parser()
    cli.description = __doc__
    cli.set_defaults(
        method='both', datasets=None,
        teacher_root=Path('/data/lthung/flow2pc/pretrained-teachers'),
        output_root=Path('/data/lthung/flow2pc/distill/student-sweep-100'),
        workers_per_gpu=1, max_parallel=8,
    )
    return cli


def selected_pairs(args):
    datasets = list(args.datasets or DATASETS)
    if len(datasets) != len(set(datasets)) or set(datasets) - set(DATASETS):
        raise ValueError('Use unique dataset names, with gom/beijing rather than recipe suffixes')
    methods = ('alicfm', 'otpfm') if args.method == 'both' else (args.method,)
    pairs = [(method, dataset) for dataset in datasets for method in methods]
    pairs = [pair for pair in pairs if pair not in PREVIOUS_SWEEPS]
    if not pairs:
        raise ValueError('All requested pairs already have previous student sweeps; nothing to schedule')
    return pairs


def validation_policy(config, dataset):
    scope = config['distillation'].get('validation_scope', 'dataset')
    if scope == 'dataset':
        scope = 'heldout_times' if dataset in ('gom', 'beijing') else 'observed_rows'
    if scope not in ('heldout_times', 'observed_rows'):
        raise ValueError(f'Unknown validation scope: {scope}')
    tracking = dataset == 'cell_tracking'
    return {
        'scope': scope,
        'selection_metric': ('worst_global_late_final_velocity_mse_over_teacher_energy'
                             if tracking else 'validation_student_teacher_velocity_mse'),
        'score_is_raw_mse': not tracking,
        'note': ('Existing tracking score: max(global, late, final) MSE / fixed teacher energy; '
                 'the same teacher, seed and validation settings are used across configurations'
                 if tracking else 'Existing dataset-specific validation velocity MSE'),
        'population_emd_used_for_selection': False,
        'uq_coverage_used_for_selection': False,
    }


def base_runs(args):
    bases = []
    for method, dataset in selected_pairs(args):
        child = copy.copy(args)
        child.method = method
        child.datasets = [f'{dataset}_w2' if method == 'otpfm' and dataset in ('gom', 'beijing') else dataset]
        child.teacher_root = args.teacher_root / method if args.method == 'both' else args.teacher_root
        child.output_root = args.output_root / method
        run = batch.build_runs(child)[0]
        # Run.dataset is a common dataset label; teacher recipe remains in config.
        run.dataset = dataset
        run.teacher_sha256 = batch.teacher_identity(run.config)[1]
        if run.config['student']['name'] != 'hclt' or not run.config['evaluation']['enabled']:
            raise ValueError('Student sweeps require an HCLT preset with final evaluation/images enabled')
        bases.append(run)
    return bases


def build_runs(args):
    for values in (args.latents, args.time_features, args.learning_rates):
        if not values or len(values) != len(set(values)):
            raise ValueError('Sweep axes must be nonempty and unique')
    if (any(v < 2 for v in args.latents)
            or any(v < 2 or v % 2 for v in args.time_features)
            or any(not math.isfinite(v) or v <= 0 for v in args.learning_rates)
            or (args.steps is not None and args.steps < 1)):
        raise ValueError('Invalid HCLT sizes, time features, learning rates or steps')
    if args.teacher_root is None:
        raise ValueError('An explicit --teacher-root is required for the current frozen teachers')
    bases = base_runs(args)
    combinations = list(itertools.product(args.latents, args.time_features, args.learning_rates))
    random.Random(2026).shuffle(combinations)
    plans = []
    for base in bases:
        params, settings = base.config['student']['params'], base.config['distillation']
        baseline = (params['num_latents'], params['time_embedding_dim'], settings['lr'])
        ordered = list(combinations)
        if baseline in ordered:
            ordered.remove(baseline)
            ordered.insert(0, baseline)
        plans.append(ordered)
    runs = []
    # Round-robin across datasets/methods: every baseline is covered first.
    for position in range(len(combinations)):
        for base, ordered in zip(bases, plans, strict=True):
            latents, features, lr = ordered[position]
            config = copy.deepcopy(base.config)
            method = config['teacher'].get('name', 'alicfm')
            config['student']['params'].update(num_latents=latents, time_embedding_dim=features)
            settings = config['distillation']
            settings.update(lr=lr, device='cuda:0', resume=True)
            if args.steps is not None:
                settings['steps'] = args.steps
            scientific = {'config': batch.scientific_config(config), 'teacher_sha256': base.teacher_sha256}
            signature = hashlib.sha256(json.dumps(scientific, sort_keys=True, allow_nan=False).encode()).hexdigest()[:12]
            name = (f'{base.dataset}-{method}-hclt-h{latents}-f{features}-lr{lr:g}'
                    f'-b{settings["batch_size"]}-n{settings["steps"]}-s{config["seed"]}-{signature}')
            output = args.output_root / method / base.dataset / name
            config['output']['dir'] = str(output)
            config['logging'].update(project='flow2pc', mode=args.mode, name=name)
            run = batch.Run(base.dataset, name, config, output)
            run.teacher_sha256 = base.teacher_sha256
            runs.append(run)
    if len({r.output for r in runs}) != len(runs):
        raise ValueError('Duplicate sweep output paths')
    return runs


def result_row(run):
    row = sweep.result_row(run)
    policy = validation_policy(run.config, run.dataset)
    score = row.pop('validation_velocity_mse')
    row.update(method=run.config['teacher'].get('name', 'alicfm'),
               teacher_sha256=run.teacher_sha256,
               validation_scope=policy['scope'], selection_metric=policy['selection_metric'],
               validation_selection_score=score,
               validation_velocity_mse=score if policy['score_is_raw_mse'] else None)
    return row


def write_yaml(path, value):
    temporary = path.with_suffix('.yaml.tmp')
    try:
        temporary.write_text(yaml.safe_dump(value, sort_keys=False))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def summary(args, runs, active, state):
    rows = [result_row(run) for run in runs]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path = args.output_root / 'summary.csv'
    temporary = path.with_suffix('.csv.tmp')
    try:
        temporary.write_text(stream.getvalue())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    selected = []
    for method, dataset in sorted({(r['method'], r['dataset']) for r in rows}):
        cohort = [r for r in rows if (r['method'], r['dataset']) == (method, dataset)]
        configs = {r.name: r.config for r in runs if r.dataset == dataset
                   and r.config['teacher'].get('name', 'alicfm') == method}
        candidates = [r for r in cohort if r['status'] == 'complete'
                      and type(r['validation_selection_score']) in (int, float)
                      and math.isfinite(r['validation_selection_score'])
                      and r['validation_selection_score'] >= 0]
        candidates.sort(key=lambda r: (r['validation_selection_score'], r['run']))
        policy = validation_policy(next(iter(configs.values())), dataset)
        directory = args.output_root / method / dataset
        record = {'method': method, 'dataset': dataset, **policy,
                  'teacher_checkpoint': next(iter(configs.values()))['teacher']['checkpoint'],
                  'teacher_sha256': cohort[0]['teacher_sha256'],
                  'completed': len(candidates), 'planned': len(cohort), 'best': candidates[:10]}
        batch.atomic_json(directory / 'best_models.json', record)
        winner = {**record, 'best': candidates[:1], 'selection_final': len(candidates) == len(cohort)}
        batch.atomic_json(directory / 'selected_student.json', winner)
        if candidates:
            config_path = directory / 'best_config.yaml'
            write_yaml(config_path, configs[candidates[0]['run']])
            selected.append({'method': method, 'dataset': dataset, **policy,
                             'best_config': str(config_path), **candidates[0]})
        else:
            (directory / 'best_config.yaml').unlink(missing_ok=True)
    batch.atomic_json(args.output_root / 'selected_students.json', {
        'selection': 'dataset_validation_velocity_MSE_only',
        'teacher_training': False, 'population_emd_used_for_selection': False,
        'selected': selected,
    })
    batch.atomic_json(args.output_root / '_launcher/status.json', {
        'status': state, 'total': len(runs),
        'counts': dict(Counter(r.status for r in runs)),
        'active': [{'run': w['task'].name, 'gpu': w['device'], 'pid': w['process'].pid} for w in active],
    })


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    args.teacher_root = args.teacher_root.expanduser().resolve()
    # For one method the shared --teacher-root contract points at that method.
    if args.method != 'both' and (args.teacher_root / args.method).is_dir():
        args.teacher_root = args.teacher_root / args.method
    try:
        runs = build_runs(args)
        details = {
            'scope': 'missing_student_sweeps_only',
            'excluded_previous_sweeps': [f'{method}/{dataset}' for method, dataset in sorted(PREVIOUS_SWEEPS)],
            'runs_per_pair': dict(Counter(f'{r.config["teacher"].get("name", "alicfm")}/{r.dataset}' for r in runs)),
            'teachers': {f'{r.config["teacher"].get("name", "alicfm")}/{r.dataset}':
                         {'checkpoint': r.config['teacher']['checkpoint'], 'sha256': r.teacher_sha256}
                         for r in runs},
            'validation_protocols': {f'{r.config["teacher"].get("name", "alicfm")}/{r.dataset}':
                                     validation_policy(r.config, r.dataset) for r in runs},
        }
        return sweep.execute(args, runs, report=summary, details=details)
    except (ValueError, OSError, RuntimeError, KeyError, yaml.YAMLError) as error:
        cli.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
