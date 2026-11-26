"""Student sweep contracts; execute only with explicit test authorization."""

import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sweep(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('_student_sweep', ROOT / 'scripts/run_student_sweep.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    # Metadata-only teacher stand-ins; never require the research filesystem.
    monkeypatch.setattr(module.batch, 'read_json', lambda path: {
        'method': path.parent.parent.name, 'checkpoint_relpath': 'checkpoints/last.pt',
    })
    monkeypatch.setattr(module.batch, 'teacher_identity', lambda config: (
        config['teacher'].get('name', 'alicfm'),
        str(Path(config['teacher']['checkpoint']).parent.parent), None, None,
    ))
    return module


def options(sweep, tmp_path, *extra):
    return sweep.parser().parse_args([
        '--output-root', str(tmp_path / 'output'), '--teacher-root', str(tmp_path / 'teachers'),
        *extra,
    ])


def test_default_covers_only_19_missing_sweeps_without_writing_files(sweep, tmp_path):
    args = options(sweep, tmp_path)
    runs = sweep.build_runs(args)
    counts = Counter((r.config['teacher']['name'], r.dataset) for r in runs)
    assert len(runs) == 1900 and len(counts) == 19 and set(counts.values()) == {100}
    assert not set(counts) & sweep.PREVIOUS_SWEEPS
    assert ('otpfm', 'cell_tracking') in counts
    assert len({r.name for r in runs}) == len({r.output for r in runs}) == 1900
    assert not (tmp_path / 'output').exists()
    for run in runs:
        method = run.config['teacher']['name']
        assert run.output.is_relative_to(args.output_root / method / run.dataset)
        recipe = f'{run.dataset}_w2' if method == 'otpfm' and run.dataset in ('gom', 'beijing') else run.dataset
        assert run.config['teacher']['checkpoint'] == str(args.teacher_root / method / recipe / 'checkpoints/last.pt')
        assert run.config['logging']['project'] == 'flow2pc'
    assert [r.name for r in runs] == [r.name for r in sweep.build_runs(args)]


@pytest.mark.parametrize('method,count', [('alicfm', 8), ('otpfm', 11)])
def test_method_filters_only_include_missing_sweeps(sweep, tmp_path, method, count):
    args = options(sweep, tmp_path, '--method', method)
    pairs = sweep.selected_pairs(args)
    assert len(pairs) == count
    assert all(pair[0] == method for pair in pairs)
    assert not set(pairs) & sweep.PREVIOUS_SWEEPS


def test_explicit_dataset_filter_does_not_resweep_previous_winners(sweep, tmp_path):
    pairs = sweep.selected_pairs(options(sweep, tmp_path, '--datasets', 'cite50', 'beijing'))
    assert pairs == [('alicfm', 'beijing')]
    with pytest.raises(ValueError, match='already have previous student sweeps'):
        sweep.selected_pairs(options(sweep, tmp_path, '--datasets', 'cite50'))


def test_only_grid_axes_change_and_each_preset_control_runs_first(sweep, tmp_path):
    args = options(sweep, tmp_path, '--datasets', 'st', 'beijing')
    bases = sweep.base_runs(args)
    runs = sweep.build_runs(args)
    for run in runs:
        base = next(b for b in bases if b.dataset == run.dataset
                    and b.config['teacher']['name'] == run.config['teacher']['name'])
        config = copy.deepcopy(run.config)
        for key in ('num_latents', 'time_embedding_dim'):
            config['student']['params'][key] = base.config['student']['params'][key]
        config['distillation']['lr'] = base.config['distillation']['lr']
        assert sweep.batch.scientific_config(config) == sweep.batch.scientific_config(base.config)
        assert config['evaluation']['enabled']
    for run, base in zip(runs[:len(bases)], bases, strict=True):
        assert run.config['student'] == base.config['student']
        assert run.config['distillation']['lr'] == base.config['distillation']['lr']
    assert sweep.validation_policy(runs[0].config, 'st')['scope'] == 'observed_rows'
    assert sweep.validation_policy(runs[-1].config, 'beijing')['scope'] == 'heldout_times'


def test_teacher_identity_changes_output_namespace(sweep, tmp_path, monkeypatch):
    args = options(sweep, tmp_path, '--datasets', 'knot', '--latents', '128',
                   '--time-features', '32', '--learning-rates', '.001')
    original = sweep.build_runs(args)
    monkeypatch.setattr(sweep.batch, 'teacher_identity', lambda config: ('otpfm', 'new-hash', None, None))
    changed = sweep.build_runs(args)
    assert all(a.output != b.output for a, b in zip(original, changed, strict=True))


def test_winner_uses_mse_and_exports_exact_run_config(sweep, tmp_path):
    args = options(sweep, tmp_path, '--datasets', 'st', '--latents', '64', '128',
                   '--time-features', '32', '--learning-rates', '.001')
    runs = sweep.build_runs(args)
    for index, run in enumerate(runs):
        run.status = 'complete'
        run.state = {'best_selection_score': .01 if index < 2 else .02, 'best_step': 1000}
        # Prefer the worse MSE with EMD; the ranking must ignore it.
        run.metrics = {'student_ode_emd_heldout': 100.0 if index < 2 else 1.0}
    args.output_root.mkdir(parents=True)
    sweep.summary(args, runs, [], 'complete')
    for method in ('alicfm', 'otpfm'):
        directory = args.output_root / method / 'st'
        result = json.loads((directory / 'selected_student.json').read_text())
        assert result['population_emd_used_for_selection'] is False
        assert result['selection_final'] is True
        assert result['best'][0]['validation_velocity_mse'] == .01
        config = yaml.safe_load((directory / 'best_config.yaml').read_text())
        assert config == next(r.config for r in runs if r.name == result['best'][0]['run'])
    assert len(json.loads((args.output_root / 'selected_students.json').read_text())['selected']) == 2


def test_tracking_score_is_honestly_labeled_and_incomplete_runs_are_ineligible(sweep, tmp_path):
    args = options(sweep, tmp_path, '--datasets', 'cell_tracking', '--latents', '128', '256',
                   '--time-features', '32', '--learning-rates', '.001')
    runs = sweep.build_runs(args)
    runs[0].status = 'complete'
    runs[0].state = {'best_selection_score': .02, 'best_step': 1000}
    runs[1].status = 'running'
    runs[1].state = {'best_selection_score': .001, 'best_step': 1000}
    args.output_root.mkdir(parents=True)
    sweep.summary(args, runs, [], 'running')
    result = json.loads((args.output_root / 'otpfm/cell_tracking/selected_student.json').read_text())
    assert result['selection_final'] is False
    assert result['score_is_raw_mse'] is False
    assert result['best'][0]['validation_velocity_mse'] is None
    assert result['best'][0]['validation_selection_score'] == .02
    assert result['best'][0]['run'] == runs[0].name


def test_preview_never_writes_queries_gpus_or_launches_workers(sweep, tmp_path, monkeypatch):
    monkeypatch.setattr(sweep.batch, 'inspect_run', lambda *args: None)
    monkeypatch.setattr(sweep.batch, 'gpu_inventory', lambda: pytest.fail('GPU query'))
    monkeypatch.setattr(sweep.batch, 'DistillationQueue', lambda *a, **k: pytest.fail('worker launch'))
    assert sweep.main(['--dry-run', '--datasets', 'knot', '--output-root', str(tmp_path / 'absent'),
                       '--teacher-root', str(tmp_path / 'teachers')]) == 0
    assert not (tmp_path / 'absent').exists()


@pytest.mark.parametrize('extra', [
    ['--datasets', 'gom_w2'], ['--datasets', 'knot', 'knot'],
    ['--latents', '128', '128'], ['--time-features', '3'],
    ['--learning-rates', 'nan'], ['--steps', '0'],
])
def test_invalid_grids_and_recipe_aliases_are_rejected(sweep, tmp_path, extra):
    with pytest.raises(ValueError):
        sweep.build_runs(options(sweep, tmp_path, *extra))
