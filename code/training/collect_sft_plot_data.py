"""Read-only export of formal SFT training and fixed probe figure sources."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__import__('os').environ.get('LFM_ROOT', '/path/to/visual_trajectory'))
SOURCES = {}


def read(path):
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def main():
    data = {'schema': 1, 'root': str(ROOT), 'selection': {'streetview': 500, 'satellite': 700}}
    settings = [('streetview', 626, 62, 8, [100, 200, 300, 313, 400, 500, 600, 626]),
                ('satellite', 752, 75, 9, [100, 200, 300, 376, 400, 500, 600, 700, 752])]
    for key, final, ntrain, nval, candidates in settings:
        run = ROOT/'runs'/f'{key}_sft_qwen3vl8b_lora_v1'
        state = read(run/f'training/checkpoint-{final}/trainer_state.json')
        current = read(run/'training/trainer_state.json')
        assert state['global_step'] == current['global_step'] == final
        # The root state appends a runtime summary after final checkpoint save.
        assert [r for r in state['log_history'] if 'loss' in r or 'eval_loss' in r] == [
            r for r in current['log_history'] if 'loss' in r or 'eval_loss' in r]
        training = [{k:r[k] for k in ['step', 'loss', 'learning_rate', 'grad_norm']}
                    for r in state['log_history'] if 'loss' in r]
        validation = [{'step':r['step'], 'loss':r['eval_loss']}
                      for r in state['log_history'] if 'eval_loss' in r]
        assert len(training) == ntrain and len(validation) == nval
        assert all(math.isfinite(v) for r in training+validation for v in r.values())
        probes = []
        for step in [0]+candidates:
            name = 'baseline_validation_probe' if step == 0 else f'checkpoint-{step}'
            overall = read(run/f'vllm/evaluations/{name}/summary.json')['overall']
            assert overall['n'] == 256
            exact = round(overall['exact_match']*256)
            adjacency = round(overall['directed_adjacency_accuracy']*1792)
            assert abs(exact/256-overall['exact_match']) < 1e-9
            assert abs(adjacency/1792-overall['directed_adjacency_accuracy']) < 1e-9
            probes.append({'step':step, 'exact':exact, 'correct_edges':adjacency,
                           'em':overall['exact_match'], 'adjacency':overall['directed_adjacency_accuracy']})
        assert probes[0]['exact'] == 0
        data[key] = {'training':training, 'validation':validation, 'probes':probes, 'final_update':final}
    data['source_sha256'] = SOURCES
    print(json.dumps(data))


if __name__ == '__main__':
    main()
