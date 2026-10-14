"""Persistent two-GPU vLLM generation; frozen inputs and original strict metrics."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import time
import traceback

GPU_UUIDS = __import__('os').environ.get('LFM_GPU_UUIDS', 'GPU-PLACEHOLDER-0,GPU-PLACEHOLDER-1').split(',')
CONCURRENCY = int(os.environ.get('STREETVIEW_VLLM_CONCURRENCY', '16'))
assert CONCURRENCY in (8, 16)
PENDING_LIMIT = 32
ENGINE_SETTINGS = dict(dtype='bfloat16', tensor_parallel_size=1, max_model_len=8192,
                       max_num_seqs=CONCURRENCY, max_num_batched_tokens=8192,
                       gpu_memory_utilization=.55, enable_lora=True, max_lora_rank=64,
                       max_loras=1, max_cpu_loras=8, lora_dtype='bfloat16',
                       limit_mm_per_prompt={'image': 8, 'video': 0},
                       mm_processor_kwargs={'min_pixels': 65536, 'max_pixels': 524288,
                                            'size': {'shortest_edge': 65536, 'longest_edge': 524288}},
                       mm_processor_cache_gb=0, enable_prefix_caching=False,
                       generation_config='vllm', seed=20261006)


def worker(rank, commands, responses, directory):
    # Set visibility before importing a CUDA runtime; each engine sees one physical GPU.
    # This pinned vLLM release uses numeric NVML indices; verify the UUID before load.
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank+1)
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    log = open(Path(directory)/f'engine.rank{rank}.log', 'a', buffering=1)
    os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
    try:
        import torch
        from PIL import Image
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from streetview_sft_common import (ROOT, MODEL, DATA, atomic_json, hash_file,
                                          read_jsonl, processor, render, text_ids, inputs, score)
        torch.backends.cuda.matmul.allow_tf32 = False
        actual_uuid = 'GPU-' + str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
        assert actual_uuid == GPU_UUIDS[rank], (actual_uuid, GPU_UUIDS[rank])
        p = processor()
        prompt = json.loads((DATA/'prompt.json').read_text())
        llm = LLM(model=str(MODEL), tokenizer=str(MODEL), **ENGINE_SETTINGS)
        params = SamplingParams(temperature=0, top_p=1, top_k=-1, max_tokens=3072,
                                repetition_penalty=1, presence_penalty=0, frequency_penalty=0,
                                stop_token_ids=[151645], ignore_eos=True,
                                skip_special_tokens=True)
        atomic_json(Path(directory)/f'engine.rank{rank}.json',
                    {'rank': rank, 'pid': os.getpid(), 'uuid': GPU_UUIDS[rank],
                     'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                     'vllm': vllm.__version__, 'torch': torch.__version__,
                     'settings': ENGINE_SETTINGS, 'ready_time': time.time()})
        responses.put({'rank': rank, 'ready': True})
        while True:
            task = commands.get()
            if task is None:
                break
            output = Path(task['output']); output.mkdir(parents=True, exist_ok=True)
            selected = read_jsonl(task['manifest'])[rank::2]
            path = output/f'predictions.rank{rank}.jsonl'
            existing = read_jsonl(path) if path.exists() else []
            allowed = {r['id']: r for r in selected}
            assert len({r['id'] for r in existing}) == len(existing)
            for row in existing:
                assert row['id'] in allowed and row['engine'] == 'vllm-0.17.1'
                assert all(row[k] == allowed[row['id']][k] for k in ['city','shuffle','ground_truth'])
                assert row['protocol_sha256'] == task['protocol_sha256']
            completed = {r['id'] for r in existing}
            todo = [r for r in selected if r['id'] not in completed]
            adapter = task.get('adapter')
            request = LoRARequest(Path(adapter).name, task['lora_id'], adapter) if adapter else None
            assert not adapter or json.loads((Path(adapter)/'adapter_config.json').read_text())['r'] == 64
            pending = {}
            started_stage = time.monotonic()
            with path.open('a', encoding='utf-8') as f:
                next_record = iter(todo)
                exhausted = False
                # Pinned vLLM 0.17.1: use the same preprocessing/request path as
                # LLM.generate, but consume finished outputs and replenish the
                # bounded waiting queue without a whole-batch barrier.
                assert vllm.__version__ == '0.17.1'
                def refill():
                    nonlocal exhausted
                    while len(pending) < PENDING_LIMIT and not exhausted:
                        record = next(next_record, None)
                        if record is None:
                            exhausted = True
                            break
                        images = []
                        for rel in record['images']:
                            ipath = (ROOT/'dataset/streetview'/rel).resolve()
                            assert ipath.is_relative_to((ROOT/'dataset/streetview').resolve())
                            with Image.open(ipath) as im:
                                assert im.size == (1024,512)
                                rgb = im.convert('RGB'); images.append(rgb)
                        rendered = render(p, record, prompt, False)
                        expected = text_ids(p, rendered)
                        assert expected.count(151655) == 4096 and len(expected)+3072 <= 8192
                        # Actual HF processor check during the pilot, not just a token estimate.
                        if task.get('pilot'):
                            reference = inputs(p, record, prompt, False)
                            assert reference['input_ids'][0].tolist() == expected
                        submitted = time.monotonic()
                        ids = llm._add_completion_requests(
                            [{'prompt': rendered, 'multi_modal_data': {'image': images}}],
                            params, lora_request=request, use_tqdm=False)
                        assert len(ids) == 1
                        external_id = llm.llm_engine.output_processor.request_states[ids[0]].external_req_id
                        assert external_id not in pending
                        pending[external_id] = (record, expected, images, submitted)
                refill()
                while pending:
                    assert llm.llm_engine.has_unfinished_requests()
                    for result in llm.llm_engine.step():
                        if not result.finished:
                            continue
                        record, expected, images, submitted = pending.pop(result.request_id)
                        elapsed = time.monotonic() - submitted
                        assert result.prompt_token_ids == expected, 'vLLM/HF prompt tokens differ'
                        assert len(result.outputs) == 1
                        sequence = result.outputs[0]; tokens = list(sequence.token_ids)
                        assert tokens and len(tokens) <= 3072
                        # Decode exactly as the existing evaluator, including whitespace handling.
                        text = p.tokenizer.decode(tokens, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
                        clipped = len(tokens) >= 3072 and tokens[-1] != 151645
                        row = {'id':record['id'], 'city':record['city'],
                               'ground_truth':record['ground_truth'], 'shuffle':record['shuffle'],
                               'text':text, 'output_tokens':len(tokens), 'seconds':elapsed,
                               'seconds_definition':'request latency including waiting and preprocessing',
                               'active_concurrency_limit':CONCURRENCY,
                               'pending_request_limit':PENDING_LIMIT,
                               'finish_reason':sequence.finish_reason, 'stop_reason':sequence.stop_reason,
                               'last_token_id':tokens[-1],
                               'engine':'vllm-0.17.1', 'protocol_sha256':task['protocol_sha256'],
                               'prompt_token_sha256':__import__('hashlib').sha256(json.dumps(expected).encode()).hexdigest(),
                               'adapter_sha256':task.get('adapter_sha256'),
                               **score(text,record['ground_truth'],prompt['step_titles'],clipped)}
                        f.write(json.dumps(row, ensure_ascii=False)+'\n'); f.flush()
                        completed.add(record['id'])
                        for im in images: im.close()
                        atomic_json(output/f'progress.rank{rank}.json',
                                {'completed':len(completed), 'total':len(selected),
                                 'last_id':record['id'], 'time':time.time(),
                                 'stage_seconds':time.monotonic()-started_stage})
                    refill()
                assert not llm.llm_engine.has_unfinished_requests()
            assert len(completed) == len(selected)
            responses.put({'rank':rank,'stage':task['name'],'completed':len(completed),
                           'seconds':time.monotonic()-started_stage})
    except BaseException:
        error = traceback.format_exc(); print(error, flush=True)
        responses.put({'rank':rank,'error':error})
        raise


class Engines:
    def __init__(self, directory):
        ctx = mp.get_context('spawn')
        self.responses = ctx.Queue()
        self.commands = [ctx.Queue() for _ in range(2)]
        self.processes = [ctx.Process(target=worker,args=(r,self.commands[r],self.responses,str(directory))) for r in range(2)]
        for child in self.processes: child.start()
        from streetview_sft_common import atomic_json
        atomic_json(Path(directory)/'engines.pid.json', {'parent':os.getpid(),
                    'workers':[p.pid for p in self.processes], 'started':time.time(), 'uuids':GPU_UUIDS})
        try:
            self.wait(3600, ready=True)
        except BaseException:
            self.close()
            raise

    def wait(self, timeout, ready=False, name=None):
        deadline = time.monotonic()+timeout
        received = {}
        while len(received)<2:
            assert time.monotonic()<deadline, 'vLLM stage timeout'
            try: response = self.responses.get(timeout=2)
            except queue.Empty:
                assert all(p.is_alive() for p in self.processes), 'vLLM worker exited; inspect engine logs'
                continue
            if 'error' in response: raise RuntimeError(response['error'])
            assert response['rank'] not in received
            assert response.get('ready') if ready else response['stage']==name
            received[response['rank']] = response
        return received

    def generate(self, task):
        for commands in self.commands: commands.put(task)
        return self.wait(345600, name=task['name'])

    def close(self):
        for commands in self.commands: commands.put(None)
        for child in self.processes:
            child.join(timeout=20)
            if child.is_alive(): child.terminate(); child.join(timeout=10)


def main():
    import fcntl
    import math
    from streetview_sft_common import (RUN, DATA, MODEL, ROOT, atomic_json,
                                       hash_file, read_jsonl, summarize)
    from audit_streetview_sft_result import check_evaluation, verify_formal_resume
    ap = argparse.ArgumentParser(); ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--pilot-count', type=int, default=64)
    args = ap.parse_args()
    directory_name = os.environ.get('STREETVIEW_VLLM_OUTPUT', 'vllm')
    assert directory_name in ('vllm', 'vllm_benchmark8', 'vllm_benchmark16')
    directory = RUN/directory_name; directory.mkdir(exist_ok=True)
    lock = (RUN/'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    completed = json.loads((RUN/'training/training_complete.json').read_text())
    assert completed['completed'] and completed['step']==626
    verify_formal_resume(json.loads((RUN/'training/configuration.json').read_text()))
    settings = {'engine':'vllm','version':'0.17.1','settings':ENGINE_SETTINGS,
                'scheduler':'continuous refill through pinned LLM request path',
                'pending_request_limit_per_gpu':PENDING_LIMIT,
                'greedy':True,'max_new_tokens':3072,'stop_token_id':151645,
                'native_eos_ignored':True,'images':8,'image_size':[1024,512],
                'prompt_sha256':hash_file(DATA/'prompt.json'),
                'tokenizer_sha256':hash_file(MODEL/'tokenizer.json'),
                'base_model_revision':json.loads((RUN/'training/configuration.json').read_text())['base_revision'],
                'lora_inference_precision':'bfloat16; original FP32 adapter files preserved',
                'source_hashes':{n:hash_file(ROOT/'code/training'/n) for n in
                                ['evaluate_streetview_vllm.py','audit_streetview_vllm_result.py',
                                 'streetview_sft_common.py']}}
    protocol = directory/'generation_protocol.json'
    if protocol.exists(): assert json.loads(protocol.read_text())==settings
    else: atomic_json(protocol, settings)
    digest = hash_file(protocol)
    engines = None
    def status(phase, **kwargs):
        atomic_json(RUN/'experiment_status.json', {'phase':phase,'pid':os.getpid(),
                    'engine':'vllm','time':time.time(),**kwargs})
        print(phase, json.dumps(kwargs), flush=True)
    def evaluate(name, manifest, adapter=None, lora_id=1, pilot=False):
        output = directory/('pilot' if pilot else 'evaluations')/name
        records = read_jsonl(manifest)
        task = {'name':name,'manifest':str(manifest),'output':str(output),
                'adapter':str(adapter) if adapter else None,'lora_id':lora_id,
                'adapter_sha256':hash_file(adapter/'adapter_model.safetensors') if adapter else None,
                'protocol_sha256':digest,'pilot':pilot}
        status('vllm_'+name, total=len(records))
        started = time.time(); ranks = engines.generate(task)
        rows = [row for rank in range(2) for row in read_jsonl(output/f'predictions.rank{rank}.jsonl')]
        assert len(rows)==len(records) and len({x['id'] for x in rows})==len(records)
        atomic_json(output/'summary.json',summarize(rows))
        summary = check_evaluation(output, records, json.loads((DATA/'prompt.json').read_text()))
        atomic_json(output/'stage.exit.json',{'exit_code':0,'started':started,'ended':time.time(),
                    'ranks':ranks,'manifest_sha256':hash_file(manifest),**task})
        return summary
    try:
        status('vllm_engine_initialization')
        engines = Engines(directory)
        if args.pilot:
            available = read_jsonl(DATA/'validation_probe.jsonl')
            assert 64 <= args.pilot_count <= len(available) and args.pilot_count % 2 == 0
            records = [available[i*len(available)//args.pilot_count] for i in range(args.pilot_count)]
            manifest = directory/'pilot_manifest.jsonl'
            manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
            evaluate('baseline',manifest,pilot=True)
            evaluate('checkpoint-626',manifest,RUN/'training/checkpoint-626',8,pilot=True)
            atomic_json(directory/'pilot_verified.json',{'verified':True,'time':time.time(),
                        'trajectories_per_model':len(records),'actual_processor_grid_and_prompt_tokens_verified':True,
                        'protocol_sha256':digest})
            status('vllm_pilot_complete'); return
        assert json.loads((directory/'pilot_verified.json').read_text())['verified']
        baseline = evaluate('baseline_validation_probe',DATA/'validation_probe.jsonl')
        metrics = read_jsonl(RUN/'training/metrics.jsonl'); best=None
        candidates=[]
        for index,step in enumerate([100,200,300,313,400,500,600,626],1):
            checkpoint=RUN/'training'/f'checkpoint-{step}'
            summary=evaluate(f'checkpoint-{step}',DATA/'validation_probe.jsonl',checkpoint,index)
            losses=[x['eval_loss'] for x in metrics if x['step']==step and 'eval_loss' in x]
            assert losses and math.isfinite(losses[-1])
            key=[summary['overall']['exact_match'],summary['overall']['directed_adjacency_accuracy'],-losses[-1]]
            candidate={'step':step,'checkpoint':str(checkpoint),'validation':summary,
                       'selection_key':key,'validation_loss':losses[-1],
                       'engine':'vllm','protocol_sha256':digest}
            candidates.append(candidate)
            if best is None or key>best['selection_key']: best=candidate
        atomic_json(directory/'checkpoint_selection.json',{'verified':True,'candidates':candidates,
                    'selected_step':best['step'],'training_completed_before_selection':True})
        atomic_json(RUN/'training/best_checkpoint.json',best)
        atomic_json(RUN/'deferred_checkpoint_selection.json',{'verified':True,
                    'completed_steps':[100,200,300,313,400,500,600,626],'selected_step':best['step'],
                    'training_completed_before_selection':True,'engine':'vllm','time':time.time()})
        checkpoint=Path(best['checkpoint']); lid=[100,200,300,313,400,500,600,626].index(best['step'])+1
        validation=evaluate('sft_validation_full',DATA/'validation_generation.jsonl',checkpoint,lid)
        atomic_json(RUN/'frozen_final_checkpoint.json',best)
        base_ood=evaluate('baseline_ood_full',DATA/'ood_test.jsonl')
        sft_ood=evaluate('sft_ood_full',DATA/'ood_test.jsonl',checkpoint,lid)
        report={'completed':True,'run_id':RUN.name,'engine':'vllm','protocol_sha256':digest,
                'best_checkpoint':best,'training':completed,
                'evaluations':{'baseline_validation_probe':baseline,'sft_validation_full':validation,
                               'baseline_ood_full':base_ood,'sft_ood_full':sft_ood},'time':time.time()}
        atomic_json(RUN/'experiment_result.json',report)
        status('complete',best_checkpoint=str(checkpoint))
    except BaseException:
        status('failed',error=traceback.format_exc()); raise
    finally:
        if engines: engines.close()


if __name__=='__main__':
    main()
