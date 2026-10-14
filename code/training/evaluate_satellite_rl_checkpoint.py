"""Bounded three-GPU vLLM evaluation of a saved Satellite RL adapter.

Uses immutable RL manifests and the original Satellite prompt/strict metrics.
Each engine persists across validation and OOD; only complete results pass audit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__import__('os').environ.get('LFM_ROOT', '/path/to/visual_trajectory'))
RUN = ROOT/'runs/satellite_grpo_opsd_qwen3vl8b_lora_v1_gpu023'
DATA = ROOT/'training/satellite_grpo_opsd_qwen3vl8b_lora_v1'
MODEL = ROOT/'models/satellite_sft_checkpoint700_single_round_bf16'
FORMAL = RUN/'deployment/ddp_resident/formal'
OUT = RUN/'evaluation/cycle_051_full'
ADAPTER = FORMAL/'global_step_51/actor'
GPUS = [0, 2, 3]
UUIDS = ['GPU-529b98e5-8518-ef78-4293-5c074fe02056',
         'GPU-aded96b3-1317-039c-a3b5-cf8871defa07',
         'GPU-0c8286b3-cb9b-7243-1d88-752a84460579']
COUNTS = {'validation_generation': 1684, 'ood_test': 4299}


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(pid):
    q = Path('/proc')/str(pid)
    try:
        a = (q/'stat').read_text().split(') ', 1)[1].split()
        return {'pid': pid, 'uid': q.stat().st_uid, 'start_ticks': a[19], 'state': a[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(b):
    a = identity(b['pid'])
    return a and a['uid'] == b['uid'] and a['start_ticks'] == b['start_ticks'] and a['state'] not in ('Z', 'X')


def state(phase, **kw):
    value = {'phase': phase, 'time': time.time(), 'coordinator': identity(os.getpid()), **kw}
    atomic(OUT/'status.json', value)
    print(json.dumps(value), flush=True)


def protocol(common):
    import importlib.metadata as metadata
    return {'cycle': 51, 'optimizer_update': 204, 'engine': 'vllm',
            'vllm': metadata.version('vllm'), 'torch': metadata.version('torch'),
            'model': str(MODEL), 'model_config_sha256': sha(MODEL/'config.json'),
            'adapter': str(ADAPTER), 'adapter_sha256': sha(ADAPTER/'adapter_model.safetensors'),
            'prompt_sha256': sha(DATA/'prompt.json'),
            'split_sha256': {k: sha(DATA/(k+'.jsonl')) for k in COUNTS},
            'gpus': GPUS, 'tp': 1, 'dtype': 'bfloat16', 'lora_dtype': 'bfloat16',
            'temperature': 0, 'top_p': 1, 'top_k': -1, 'n': 1,
            'max_response_tokens': 3072, 'stop_token_ids': [151645], 'ignore_eos': True,
            'repetition_penalty': 1, 'presence_penalty': 0, 'frequency_penalty': 0,
            'privileged_gt_in_student_prompt': False, 'image_size': [768, 768],
            'min_pixels': 65536, 'max_pixels': 524288, 'max_model_len': 9216,
            'max_num_seqs': 16, 'max_num_batched_tokens': 8192,
            'submission_batch': 96, 'gpu_memory_utilization': .60,
            'enable_prefix_caching': False, 'mm_processor_cache_gb': 0, 'seed': 20261009}


def worker(rank):
    import torch
    import satellite_sft_common as common
    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    gpu = GPUS[rank]
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu)
    actual = 'GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    assert actual == UUIDS[rank], (actual, UUIDS[rank])
    torch.backends.cuda.matmul.allow_tf32 = False
    p = common.processor()
    prompt = json.loads((DATA/'prompt.json').read_text())
    contract = json.loads((OUT/'protocol.json').read_text())
    assert protocol(common) == contract
    ph = sha(OUT/'protocol.json')
    atomic(OUT/f'worker_{gpu}_identity.json', {**identity(os.getpid()), 'gpu': gpu,
           'uuid': actual, 'cuda_visible_devices': str(gpu), 'phase': 'loading_engine'})
    llm = LLM(model=str(MODEL), tokenizer=str(common.MODEL), dtype='bfloat16',
              tensor_parallel_size=1, max_model_len=9216, max_num_seqs=16,
              max_num_batched_tokens=8192, gpu_memory_utilization=.60,
              enable_lora=True, max_lora_rank=64, max_loras=1, lora_dtype='bfloat16',
              limit_mm_per_prompt={'image': 8, 'video': 0},
              mm_processor_kwargs={'min_pixels': 65536, 'max_pixels': 524288,
                                   'size': {'shortest_edge': 65536, 'longest_edge': 524288}},
              mm_processor_cache_gb=0, enable_prefix_caching=False,
              generation_config='vllm', seed=20261009)
    params = SamplingParams(temperature=0, top_p=1, top_k=-1, n=1, max_tokens=3072,
              repetition_penalty=1, presence_penalty=0, frequency_penalty=0,
              stop_token_ids=[151645], ignore_eos=True)
    request = LoRARequest('satellite_cycle51', 51, str(ADAPTER))
    atomic(OUT/f'worker_{gpu}_identity.json', {**identity(os.getpid()), 'gpu': gpu,
           'uuid': actual, 'cuda_visible_devices': str(gpu), 'phase': 'engine_ready'})
    for split, total in COUNTS.items():
        records = common.read_jsonl(DATA/(split+'.jsonl'))
        assert len(records) == total
        selected = records[rank::len(GPUS)]
        directory = OUT/split/f'gpu_{gpu}'
        directory.mkdir(parents=True, exist_ok=True)
        assert not (directory/'predictions.jsonl').exists(), 'No automatic rerun or overwrite'
        completed = 0
        started = time.monotonic()
        common.atomic_json(directory/'progress.json', {'completed': 0, 'total': len(selected),
                          'gpu': gpu, 'phase': 'generating', 'time': time.time()})
        with (directory/'predictions.jsonl').open('w', encoding='utf-8') as stream:
            for start in range(0, len(selected), 96):
                chunk = selected[start:start+96]
                requests, expected_ids, all_images = [], [], []
                try:
                    for i, record in enumerate(chunk):
                        images = []
                        for relative in record['images']:
                            path = (ROOT/'dataset/satellite'/relative).resolve()
                            assert path.is_relative_to((ROOT/'dataset/satellite').resolve())
                            with Image.open(path) as im:
                                assert im.size == (768, 768)
                                images.append(im.convert('RGB'))
                        text = common.render(p, record, prompt, False)
                        expected = common.text_ids(p, text)
                        assert expected.count(151655) == 3872 and len(expected) <= 5120
                        if start == 0 and i == 0:
                            assert common.inputs(p, record, prompt, False)['input_ids'][0].tolist() == expected
                        expected_ids.append(expected)
                        requests.append({'prompt': text, 'multi_modal_data': {'image': images}})
                        all_images.extend(images)
                    results = llm.generate(requests, params, lora_request=request, use_tqdm=False)
                    assert len(results) == len(chunk)
                    for record, expected, result in zip(chunk, expected_ids, results, strict=True):
                        assert result.prompt_token_ids == expected, 'Frozen processor/prompt mismatch'
                        assert len(result.outputs) == 1
                        sequence = result.outputs[0]
                        tokens = list(sequence.token_ids)
                        assert 1 <= len(tokens) <= 3072
                        text = p.tokenizer.decode(tokens, skip_special_tokens=True)
                        clipped = len(tokens) == 3072 and tokens[-1] != 151645
                        row = {**{k: record[k] for k in ('id', 'city', 'shuffle', 'ground_truth')},
                               'text': text, 'output_token_ids': tokens, 'output_tokens': len(tokens),
                               'prompt_token_sha256': hashlib.sha256(json.dumps(expected).encode()).hexdigest(),
                               'protocol_sha256': ph, 'adapter_sha256': contract['adapter_sha256'],
                               'finish_reason': sequence.finish_reason, 'stop_reason': sequence.stop_reason,
                               **common.score(text, record['ground_truth'], prompt['step_titles'], clipped)}
                        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                    stream.flush()
                    completed += len(chunk)
                finally:
                    for image in all_images:
                        image.close()
                common.atomic_json(directory/'progress.json', {'completed': completed,
                    'total': len(selected), 'gpu': gpu, 'phase': 'generating',
                    'seconds': time.monotonic()-started, 'time': time.time()})
                print(f'{split} GPU{gpu}: {completed}/{len(selected)}', flush=True)
        actual_rows = common.read_jsonl(directory/'predictions.jsonl')
        assert len(actual_rows) == len({r['id'] for r in actual_rows}) == len(selected)
        common.atomic_json(directory/'summary.json', common.summarize(actual_rows))
        common.atomic_json(directory/'progress.json', {'completed': completed,
            'total': len(selected), 'gpu': gpu, 'phase': 'complete',
            'seconds': time.monotonic()-started, 'time': time.time()})


def verify_split(split, tokenizer, common):
    expected_rows = common.read_jsonl(DATA/(split+'.jsonl'))
    expected = {r['id']: r for r in expected_rows}
    assert len(expected) == len(expected_rows) == COUNTS[split]
    rows = []
    for gpu in GPUS:
        path = OUT/split/f'gpu_{gpu}'
        shard = common.read_jsonl(path/'predictions.jsonl')
        assert common.summarize(shard) == json.loads((path/'summary.json').read_text())
        rows.extend(shard)
    assert len(rows) == len({r['id'] for r in rows}) == len(expected)
    assert {r['id'] for r in rows} == set(expected)
    prompt = json.loads((DATA/'prompt.json').read_text())
    ph = sha(OUT/'protocol.json')
    ah = sha(ADAPTER/'adapter_model.safetensors')
    for row in rows:
        source = expected[row['id']]
        assert all(row[k] == source[k] for k in ('city', 'shuffle', 'ground_truth'))
        assert row['protocol_sha256'] == ph and row['adapter_sha256'] == ah
        ids = row['output_token_ids']
        assert 1 <= row['output_tokens'] == len(ids) <= 3072
        assert tokenizer.decode(ids, skip_special_tokens=True) == row['text']
        rendered = common.render(tokenizer_processor, source, prompt, False)
        prompt_sha = hashlib.sha256(json.dumps(common.text_ids(tokenizer_processor, rendered)).encode()).hexdigest()
        assert row['prompt_token_sha256'] == prompt_sha
        rescored = common.score(row['text'], source['ground_truth'], prompt['step_titles'],
                                len(ids) == 3072 and ids[-1] != 151645)
        assert all(row[k] == v for k, v in rescored.items())
    by_id = {r['id']: r for r in rows}
    ordered = [by_id[r['id']] for r in expected_rows]
    target = OUT/split/'predictions.jsonl'
    assert not target.exists()
    target.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in ordered), encoding='utf-8')
    summary = common.summarize(common.read_jsonl(target))
    common.atomic_json(OUT/split/'summary.json', summary)
    report = {'status': 'passed', 'n': len(rows), 'unique_ids': len(by_id),
              'split': split, 'cycle': 51, 'optimizer_update': 204,
              'raw_answers_rescored': True, 'token_ids_decoded': True,
              'protocol_sha256': ph, 'adapter_sha256': ah,
              'split_sha256': sha(DATA/(split+'.jsonl')),
              'prompt_sha256': sha(DATA/'prompt.json'),
              'predictions_sha256': sha(target), 'summary': summary}
    common.atomic_json(OUT/split/'report.json', report)
    return report


def coordinator():
    global tokenizer_processor
    import satellite_sft_common as common
    import audit_final
    assert not OUT.exists(), 'Inspect prior evaluation instead of launching a duplicate'
    assert json.loads((RUN/'provision/controlled_stop_for_cycle51_evaluation.json').read_text())['status'] == 'stopped_by_user_request'
    stopped = json.loads((RUN/'provision/controlled_stop_for_cycle51_evaluation.json').read_text())
    assert not any(alive(b) for b in stopped['bound_processes'])
    OUT.mkdir(parents=True)
    state('checking_checkpoint_and_inputs')
    cp = audit_final.audit_checkpoint(51)
    assert cp['adapter_sha256'] == 'f8d7975ef4f4254957b4af7a3e5bca985c26750d5632d3551420c818e36ed84e'
    contract = protocol(common)
    atomic(OUT/'protocol.json', contract)
    frozen = json.loads((RUN/'provision/frozen_inputs.json').read_text())
    assert all(sha(p) == h for p, h in frozen.items())
    sources = {**frozen, str(Path(__file__).resolve()): sha(__file__),
               str(ROOT/'code/training/satellite_sft_common.py'): sha(ROOT/'code/training/satellite_sft_common.py'),
               **{str(DATA/(k+'.jsonl')): sha(DATA/(k+'.jsonl')) for k in COUNTS}}
    atomic(OUT/'frozen_inputs.json', sources)
    atomic(OUT/'checkpoint_verification.json', cp)
    processes, handles, bindings = [], [], []
    try:
        for rank, gpu in enumerate(GPUS):
            free = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                  '--format=csv,noheader,nounits'], text=True, timeout=8)
            assert int(free.strip()) >= 55*1024, f'GPU{gpu} lacks free memory'
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu),
                       VLLM_WORKER_MULTIPROC_METHOD='spawn', TOKENIZERS_PARALLELISM='false')
            handle = (OUT/f'worker_{gpu}.log').open('ab', buffering=0)
            handles.append(handle)
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(rank)],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                    start_new_session=True)
            processes.append(proc)
            bindings.append({**identity(proc.pid), 'gpu': gpu, 'rank': rank})
        state('generating_validation_then_ood', workers=bindings, counts=COUNTS)
        deadline = time.monotonic()+8*3600
        while any(p.poll() is None for p in processes):
            failed = [(b['gpu'], p.returncode) for b, p in zip(bindings, processes) if p.poll() not in (None, 0)]
            assert not failed, f'Nonzero evaluator exit: {failed}'
            assert time.monotonic() < deadline, 'Bounded eight-hour evaluation deadline'
            time.sleep(15)
        exits = [{'gpu': b['gpu'], 'identity': b, 'exit_code': p.returncode} for b, p in zip(bindings, processes)]
        atomic(OUT/'worker_exits.json', exits)
        assert all(p.returncode == 0 for p in processes)
        state('auditing_full_results', workers=bindings)
        tokenizer_processor = common.processor()
        results = {split: verify_split(split, tokenizer_processor.tokenizer, common) for split in COUNTS}
        assert protocol(common) == contract and all(sha(p) == h for p, h in sources.items())
        cp_after = audit_final.audit_checkpoint(51)
        assert cp_after == cp
        acceptance = {'status': 'passed', 'cycle': 51, 'optimizer_update': 204,
                      'training_stopped_by_user': True, 'full_training_epoch_completed': False,
                      'worker_exits': exits, 'checkpoint': cp_after, 'results': results,
                      'frozen_inputs_unchanged': True, 'time': time.time()}
        atomic(OUT/'acceptance.json', acceptance)
        state('complete', result=str(OUT/'acceptance.json'))
    except BaseException as error:
        state('needs_attention', error=repr(error), workers=bindings)
        for b, proc in zip(bindings, processes):
            if proc.poll() is None and alive(b):
                os.killpg(proc.pid, signal.SIGTERM)
        for proc in processes:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, choices=range(3))
    args = parser.parse_args()
    if args.worker is None:
        coordinator()
    else:
        try:
            worker(args.worker)
        except BaseException:
            traceback.print_exc()
            raise
