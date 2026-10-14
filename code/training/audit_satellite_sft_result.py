"""Independently verify fresh-base Satellite training and every generation row."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import math
from pathlib import Path
import time

from satellite_sft_common import *

GPU_UUIDS = __import__('os').environ.get('LFM_GPU_UUIDS', 'GPU-PLACEHOLDER-0,GPU-PLACEHOLDER-1').split(',')
CODE_FILES = ['satellite_sft_common.py', 'export_satellite_sft.py', 'train_satellite_sft.py',
              'gpu_preflight_satellite.py', 'verify_satellite_preflight_resume.py',
              'evaluate_satellite_vllm.py', 'audit_satellite_sft_result.py',
              'verify_satellite_checkpoint_recovery.py', 'verify_satellite_callback_workflow.py',
              'run_satellite_sft.py']


def verify_model_source():
    manifest_path = ROOT/'evaluation/sft_model_source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['repo'] == 'Qwen/Qwen3-VL-8B-Instruct'
    assert manifest['revision'] == '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
    declared = json.loads((MODEL/'model_verification.json').read_text())
    assert declared['verified'] and declared['revision'] == manifest['revision']
    actual = {}
    for entry in manifest['files']:
        name = entry['rfilename']
        if name == '.gitattributes':
            continue
        path = MODEL/name
        assert path.is_file() and path.stat().st_size == entry['size']
        if 'lfs' in entry:
            digest = hash_file(path)
            assert digest == entry['lfs']['sha256']
            algorithm = 'sha256'
        else:
            b = path.read_bytes()
            digest = hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
            assert digest == entry['blobId']
            algorithm = 'git-blob-sha1'
        actual[name] = {'digest': digest, 'algorithm': algorithm, 'bytes': path.stat().st_size}
    assert len(actual) == 15
    return {'verified': True, 'revision': manifest['revision'], 'files': actual,
            'manifest_sha256': hash_file(manifest_path)}


def verify_data():
    export = json.loads((DATA/'export_verification.json').read_text())
    counts = {'train':12021, 'validation_loss':1143, 'validation_generation':1684,
              'validation_probe':256, 'ood_test':4299}
    assert export['verified'] and export['counts'] == counts
    assert export['max_steps_two_epochs'] == 752
    assert export['image_headers_verified'] == 187480
    assert export['source_image_size'] == [768,768] and export['visual_tokens_per_trajectory'] == 3872
    cot = ROOT/'dataset/satellite/cot/satellite_cot_qwen38_flash_all_20261004'
    for name, digest in export['source_hashes'].items():
        assert hash_file(cot/name) == digest, ('Frozen Satellite source changed',name)
    prompt = json.loads((DATA/'prompt.json').read_text())
    assert hash_file(DATA/'prompt.json') == export['prompt_sha256']
    assert 'satellite' in prompt['system'].lower()
    records = {}
    for name, n in counts.items():
        assert hash_file(DATA/f'{name}.jsonl') == export['derived_hashes'][name]
        rows = read_jsonl(DATA/f'{name}.jsonl')
        assert len(rows) == len({r['id'] for r in rows}) == n
        records[name] = rows
        for row in rows:
            assert len(row['images']) == len(set(row['images'])) == 8
            assert sorted(row['shuffle']) == list(range(8))
            assert row['ground_truth'] == sorted(range(8),key=row['shuffle'].__getitem__)
            for rel in row['images']:
                path = (ROOT/'dataset/satellite'/rel).resolve()
                assert path.is_relative_to((ROOT/'dataset/satellite').resolve()) and path.is_file()
    train, dev, ood = ({x['id'] for x in records[n]} for n in ['train','validation_generation','ood_test'])
    assert not (train & dev or train & ood or dev & ood)
    assert {x['city'] for x in records['train']}.isdisjoint(DEV_CITIES | OOD_CITIES)
    assert {x['city'] for x in records['validation_generation']} == DEV_CITIES
    assert {x['city'] for x in records['ood_test']} == OOD_CITIES
    assert {x['id'] for x in records['validation_loss']} <= dev
    assert {x['id'] for x in records['validation_probe']} <= dev
    assert Counter(x['city'] for x in records['validation_probe']) == {'toronto':128,'vienna':128}
    assert Counter(x['city'] for x in records['validation_generation']) == {'toronto':931,'vienna':753}
    assert Counter(x['city'] for x in records['validation_loss']) == {'toronto':581,'vienna':562}
    frozen_split = json.loads((cot/'split_manifest.json').read_text())
    assert {x['id'] for x in frozen_split['sequences'] if x['split']=='ood_test'} == ood
    p = processor()
    for record in records['train'] + records['validation_loss']:
        result = score(record['content'],record['ground_truth'],prompt['step_titles'])
        assert result['exact_match'] and result['cot_format_valid']
        assert len(text_ids(p,render(p,record,prompt,True))) == record['input_length'] <= 8192
    return export, records, prompt


def check_summary(actual, expected):
    assert actual.keys() == expected.keys()
    assert actual['invalid_score_policy'] == expected['invalid_score_policy']
    for a,b in [(actual['overall'],expected['overall']),
                (actual['city_macro_average'],expected['city_macro_average'])]:
        assert a.keys() == b.keys()
        for key,value in b.items():
            assert math.isfinite(a[key]) and math.isclose(a[key],value,rel_tol=1e-10,abs_tol=1e-10)
    assert actual['per_city'].keys() == expected['per_city'].keys()
    for city,b in expected['per_city'].items():
        a = actual['per_city'][city]
        assert a.keys() == b.keys()
        for key,value in b.items():
            assert math.isfinite(a[key]) and math.isclose(a[key],value,rel_tol=1e-10,abs_tol=1e-10)


def check_evaluation(directory, records, prompt):
    directory = Path(directory)
    expected = {r['id']:r for r in records}
    rows = []
    for rank in range(2):
        part = read_jsonl(directory/f'predictions.rank{rank}.jsonl')
        assert {r['id'] for r in part} == {r['id'] for r in records[rank::2]}
        rows += part
    assert len(rows) == len(expected) == len({r['id'] for r in rows})
    for row in rows:
        source = expected[row['id']]
        assert all(row[k]==source[k] for k in ['city','shuffle','ground_truth'])
        assert type(row['output_tokens']) is int and 0 < row['output_tokens'] <= 3072
        assert row['truncated'] == (row['output_tokens']>=3072 and row['last_token_id']!=151645)
        if row['finish_reason']=='stop':
            assert row['last_token_id']==151645 and row['stop_reason']==151645
        else:
            assert row['finish_reason']=='length' and row['output_tokens']==3072
        recomputed = score(row['text'],source['ground_truth'],prompt['step_titles'],row['truncated'])
        assert all(row[k]==v for k,v in recomputed.items())
    actual = json.loads((directory/'summary.json').read_text())
    check_summary(actual,summarize(rows))
    return actual


def verify_training_contract(config):
    completed = json.loads((RUN/'training/training_complete.json').read_text())
    assert completed['completed'] and completed['step']==752 and completed['epoch']>=1.999
    expected = {'run_id':RUN_ID,'smoke':False,'base_model':str(MODEL), 'rank':64,'alpha':128,
                'dropout':.05,'learning_rate':5e-5,'micro_batch':1,'gradient_accumulation':16,
                'world_size':2,'epochs':2,'max_steps':752,'seed':SEED,'min_pixels':65536,
                'max_pixels':524288,'max_sequence':8192,'image_size':[768,768],
                'images_per_trajectory':8,'assistant_only_loss':True,'target_field':'content',
                'frozen_visual_encoder':True,'target_modules':TARGET_MODULES,
                'train_count':12021,'validation_loss_count':1143,
                'generation_validation_schedule':'after_training','visual_tokens_per_trajectory':3872}
    assert all(config[k]==v for k,v in expected.items())
    assert config['resume_from'] is None, 'Fresh independent Satellite run expected; audit any actual resume separately'
    assert config['base_revision']=='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
    saved=torch.load(RUN/'training/checkpoint-752/training_args.bin',map_location='cpu',weights_only=False)
    actual_args={'bf16':True,'fp16':False,'per_device_train_batch_size':1,
                 'per_device_eval_batch_size':1,'gradient_accumulation_steps':16,
                 'learning_rate':5e-5,'num_train_epochs':2,'max_steps':752,'warmup_ratio':.03,
                 'adam_beta1':.9,'adam_beta2':.999,'adam_epsilon':1e-8,'weight_decay':.01,
                 'max_grad_norm':1.,'gradient_checkpointing':True,'tf32':True,
                 'remove_unused_columns':False,'ddp_find_unused_parameters':False,
                 'ddp_timeout':345600,'seed':SEED,'data_seed':SEED,'save_only_model':False,
                 'save_steps':100,'eval_steps':100}
    assert all(getattr(saved,k)==v for k,v in actual_args.items())
    assert saved.gradient_checkpointing_kwargs=={'use_reentrant':False}
    for name,value in [('optim','adamw_torch'),('lr_scheduler_type','cosine'),
                       ('save_strategy','steps'),('eval_strategy','steps')]:
        assert getattr(getattr(saved,name),'value',getattr(saved,name))==value
    before = json.loads((RUN/'pretraining_verification.json').read_text())
    assert before['verified'] and before['initialization']=='original base model; no Street View adapter'
    for name,digest in before['code_hashes'].items():
        assert hash_file(ROOT/'code/training'/name)==digest
    for name,digest in config['code_hashes'].items():
        assert hash_file(ROOT/'code/training'/name)==digest
    assert json.loads((RUN/'formal_training.exit.json').read_text())['exit_code']==0
    parameters = json.loads((RUN/'training/trainable_parameters.json').read_text())
    assert len(parameters['matched_modules'])==252
    assert len(parameters['parameters'])==504 and sum(parameters['parameters'].values())==174587904
    assert all('language_model.layers.' in n and 'lora_' in n for n in parameters['parameters'])
    memory = {}
    for rank in range(2):
        placement = json.loads((RUN/f'training/gpu_placement.rank{rank}.json').read_text())
        assert placement['verified'] and placement['gpu_uuid']==GPU_UUIDS[rank]
        row = json.loads((RUN/f'training/progress.rank{rank}.json').read_text())
        assert row['step']==752 and row['gradient_checks']==750 and row['min_nonzero_adapter_tensors']==504
        memory[str(rank)]=row
    return {'completed':completed,'memory':memory,'configuration':config}


def verify_preflight():
    workflow=json.loads((RUN/'callback_workflow_verification.json').read_text())
    assert workflow['verified'] and workflow['temporary_files_removed']
    for name,digest in workflow['code_hashes'].items():
        assert hash_file(ROOT/'code/training'/name)==digest
    for case in workflow['cases']:
        assert case['generation_calls']==0 and case['final_step']==6 and case['final_epoch']>=1.999
        steps=[6] if case['resume_from_step'] is not None else [3,6]
        assert sorted({x['step'] for x in case['saves']})==steps
        assert case['eval_steps']==([] if case['smoke'] else steps)
        assert all(x['fresh_eval']==(not case['smoke']) for x in case['saves'])
    smoke = json.loads((RUN/'preflight/preflight_complete.json').read_text())
    assert smoke['passed'] and smoke['step']==26 and smoke['resume_from']==str(RUN/'preflight/checkpoint-25')
    resume = json.loads((RUN/'resume_state_verification.json').read_text())
    assert resume['verified'] and resume['trainer_steps']==[25,26]
    assert resume['scheduler_last_epoch']==[25,26]
    assert resume['adapter_tensors_identical_after_zero_lr_resumed_update']==504
    assert resume['passed_gradient_checks']>0
    for rank in range(2):
        assert json.loads((RUN/f'gpu_preflight.rank{rank}.json').read_text())['passed']
        assert resume['gpu_placement'][rank]['gpu_uuid']==GPU_UUIDS[rank]
    for stage in ['gpu_preflight','preflight','preflight_resume','resume_state_verification']:
        assert json.loads((RUN/f'{stage}.exit.json').read_text())['exit_code']==0
    return resume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pretraining',action='store_true')
    ap.add_argument('--data-only',action='store_true')
    args = ap.parse_args()
    if args.data_only:
        export,records,prompt = verify_data()
        proof = {'verified':True,'time':time.time(),'counts':export['counts'],
                 'source_hashes':export['source_hashes'],'derived_hashes':export['derived_hashes'],
                 'prompt_sha256':export['prompt_sha256'],
                 'all_supervised_targets_and_expanded_token_lengths_recomputed':True,
                 'train_dev_ood_disjoint':True,'all_frozen_ood_ids_covered':True,
                 'max_sequence':export['max_sequence'],'mean_sequence':export['mean_sequence'],
                 'audit_source_sha256':hash_file(Path(__file__))}
        atomic_json(RUN/'data_independent_verification.json',proof)
        print('SATELLITE_DATA_INDEPENDENTLY_VERIFIED',export['counts'],flush=True)
        return
    model = verify_model_source()
    export,records,prompt = verify_data()
    resume = verify_preflight()
    if args.pretraining:
        assert not list((RUN/'training').glob('checkpoint-*'))
        versions = {n:importlib.metadata.version(n) for n in ['torch','transformers','peft','accelerate','tokenizers','pillow']}
        proof = {'verified':True,'time':time.time(),'model':model,'export':export,'resume':resume,
                 'versions':versions,'initialization':'original base model; no Street View adapter',
                 'code_hashes':{n:hash_file(ROOT/'code/training'/n) for n in CODE_FILES}}
        atomic_json(RUN/'pretraining_verification.json',proof)
        print('SATELLITE_PRETRAINING_VERIFIED',flush=True)
        return
    assert json.loads((RUN/'experiment_status.json').read_text())['phase']=='complete'
    report = json.loads((RUN/'experiment_result.json').read_text())
    assert report['completed'] and report['run_id']==RUN_ID
    contract = verify_training_contract(json.loads((RUN/'training/configuration.json').read_text()))
    protocol_path = RUN/'vllm/generation_protocol.json'
    protocol = json.loads(protocol_path.read_text()); digest=hash_file(protocol_path)
    assert report['protocol_sha256']==digest and protocol['version']=='0.17.1'
    assert protocol['images']==8 and protocol['image_size']==[768,768]
    assert protocol['prompt_sha256']==hash_file(DATA/'prompt.json')
    assert protocol['max_new_tokens']==3072 and protocol['stop_token_id']==151645 and protocol['native_eos_ignored']
    assert protocol['settings']['max_num_seqs']==8 and protocol['pending_request_limit_per_gpu']==32
    for name,value in protocol['source_hashes'].items():
        assert hash_file(ROOT/'code/training'/name)==value
    p=processor(); expected_ids=text_ids(p,render(p,records['validation_probe'][0],prompt,False))
    expected_prompt_hash=hashlib.sha256(json.dumps(expected_ids).encode()).hexdigest()
    assert expected_ids.count(151655)==3872
    max_steps=752; steps=sorted({*range(100,max_steps+1,100),max_steps//2,max_steps})
    metrics=read_jsonl(RUN/'training/metrics.jsonl')
    assert all(math.isfinite(row[k]) for row in metrics for k in ['loss','eval_loss','grad_norm'] if k in row)
    best=None; candidates=[]; evaluations={}; stage_timings={}
    pairs=[(f'checkpoint-{s}','validation_probe') for s in steps]
    pairs += [('baseline_validation_probe','validation_probe'),('sft_validation_full','validation_generation'),
              ('baseline_ood_full','ood_test'),('sft_ood_full','ood_test')]
    frozen=json.loads((RUN/'frozen_final_checkpoint.json').read_text())
    assert frozen==json.loads((RUN/'training/best_checkpoint.json').read_text())==report['best_checkpoint']
    for name,manifest in pairs:
        directory=RUN/'vllm/evaluations'/name
        summary=check_evaluation(directory,records[manifest],prompt)
        metadata=json.loads((directory/'stage.exit.json').read_text())
        assert metadata['exit_code']==0 and metadata['started']>=contract['completed']['time']
        assert metadata['protocol_sha256']==digest
        assert Path(metadata['manifest']).resolve()==(DATA/f'{manifest}.jsonl').resolve()
        assert metadata['manifest_sha256']==hash_file(DATA/f'{manifest}.jsonl')
        adapter=Path(metadata['adapter']) if metadata['adapter'] else None
        if name.startswith('baseline'):
            assert adapter is None and metadata['adapter_sha256'] is None
        else:
            expected_step=int(name.split('-')[-1]) if name.startswith('checkpoint-') else frozen['step']
            assert adapter.resolve()==(RUN/f'training/checkpoint-{expected_step}').resolve()
            assert hash_file(adapter/'adapter_model.safetensors')==metadata['adapter_sha256']
            ac=json.loads((adapter/'adapter_config.json').read_text())
            assert ac['r']==64 and ac['lora_alpha']==128 and ac['lora_dropout']==.05
            assert ac['bias']=='none' and ac['base_model_name_or_path']==str(MODEL)
            assert ac['target_modules']==TARGET_MODULES
            assert json.loads((adapter/'trainer_state.json').read_text())['global_step']==expected_step
        rows=[r for rank in range(2) for r in read_jsonl(directory/f'predictions.rank{rank}.jsonl')]
        for row in rows:
            assert row['protocol_sha256']==digest and row['adapter_sha256']==metadata['adapter_sha256']
            assert row['prompt_token_sha256']==expected_prompt_hash and row['engine']=='vllm-0.17.1'
            assert row['active_concurrency_limit']==8 and row['pending_request_limit']==32
        wall=metadata['ended']-metadata['started']; assert wall>0
        stage_timings[name]={'wall_seconds':wall,'output_tokens_per_second':sum(r['output_tokens'] for r in rows)/wall}
        if name.startswith('checkpoint-'):
            step=int(name.split('-')[-1])
            losses=[r['eval_loss'] for r in metrics if r['step']==step and 'eval_loss' in r]
            assert losses
            exact=sum(r['prediction']==r['ground_truth'] for r in rows)
            edges=sum(sum(edge in set(zip(r['prediction'],r['prediction'][1:])) for edge in zip(r['ground_truth'],r['ground_truth'][1:])) for r in rows if r['prediction'] is not None)
            candidate={'step':step,'selection_key':[exact,edges,-losses[-1]],'validation_loss':losses[-1],'summary':summary}
            candidates.append(candidate)
            if best is None or candidate['selection_key']>best['selection_key']: best=candidate
        else:
            check_summary(report['evaluations'][name],summary);evaluations[name]=summary
    assert best['step']==frozen['step'] and best['selection_key']==frozen['selection_key']
    deferred=json.loads((RUN/'deferred_checkpoint_selection.json').read_text())
    assert deferred['verified'] and deferred['training_completed_before_selection'] and deferred['completed_steps']==steps
    assert deferred['selected_step']==best['step']
    assert json.loads((RUN/'vllm/pilot_verified.json').read_text())['verified']
    for rank in range(2):
        engine=json.loads((RUN/f'vllm/engine.rank{rank}.json').read_text())
        assert engine['uuid']==GPU_UUIDS[rank] and engine['vllm']=='0.17.1'
        assert engine['settings']==protocol['settings']
    # Full OOD is evaluated only after validation selects and freezes the model.
    freeze_time=(RUN/'frozen_final_checkpoint.json').stat().st_mtime
    for name in ['baseline_ood_full','sft_ood_full']:
        meta=json.loads((RUN/f'vllm/evaluations/{name}/stage.exit.json').read_text())
        assert meta['started']>=freeze_time
    recovery=json.loads((RUN/'retained_checkpoint_recovery_verification.json').read_text())
    retained=sorted({best['step'],*steps[-2:]})
    assert recovery['verified'] and recovery['full_recovery_steps']==retained
    for step in retained:
        assert recovery['checkpoints'][str(step)]['adapter_sha256']==hash_file(RUN/f'training/checkpoint-{step}/adapter_model.safetensors')
    proof={'verified':True,'time':time.time(),'run_id':RUN_ID,'counts':export['counts'],
           'best_step':best['step'],'selection_recomputed':best,'candidates':candidates,'training':contract,
           'model_source':model,'all_prediction_metrics_recomputed':True,'evaluations':evaluations,
           'generation_protocol':protocol,'stage_timings':stage_timings,'retained_checkpoint_recovery':recovery,
           'manual_human_review':'not performed; frozen automatic eligibility used'}
    atomic_json(RUN/'final_audit.json',proof)
    a=evaluations['baseline_ood_full']['overall'];b=evaluations['sft_ood_full']['overall']
    lines=['# Satellite Qwen3-VL-8B-Instruct SFT 结果','',
           f'从原始 base model 独立训练 2 个完整 epoch、752 次更新；最佳 checkpoint 为 {best["step"]}。',
           f'同一 4,299 条 OOD：base Exact Match {a["exact_match"]:.2%}，SFT {b["exact_match"]:.2%}，变化 {(b["exact_match"]-a["exact_match"])*100:+.2f} 个百分点。','',
           '训练 12,021 条；Toronto/Vienna loss validation 1,143 条、generation validation 1,684 条、固定 probe 256 条。',
           '每条 8 张 768×768 图片，processor grid 为 [1,44,44]，合计 3,872 个视觉 tokens。保留 Satellite 独立 prompt 与冻结呈现排列。',
           'BF16、语言层 LoRA r64/alpha128/dropout0.05、lr5e-5、有效 batch32、cosine、warmup3%；基础模型和视觉编码器冻结。',
           '全部 generation 在训练后通过 vLLM 0.17.1 运行；每卡活跃8、待处理32、持续补充请求；greedy、最多3,072 tokens、仅151645结束。','',
           '| 集合 / 城市 | N | Exact Match | Pairwise | Kendall τ | Directed adjacency | Position | 合法排列 | CoT格式 | 截断 | 平均tokens |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    def table(label,v):
        return f'| {label} | {v["n"]} | {v["exact_match"]:.2%} | {v["pairwise_accuracy"]:.2%} | {v["kendall_tau"]:.4f} | {v["directed_adjacency_accuracy"]:.2%} | {v["position_accuracy"]:.2%} | {v["valid_answer"]:.2%} | {v["cot_format_valid"]:.2%} | {v["truncated"]:.2%} | {v["output_tokens"]:.1f} |'
    for name,summary in evaluations.items():
        lines.append(table(name,summary['overall']))
        for city,v in summary['per_city'].items(): lines.append(table(f'{name} / {city}',v))
        if name.endswith('ood_full'): lines.append(table(f'{name} / city macro',{'n':4299,**summary['city_macro_average']}))
    lines += ['', '模型选择仅使用固定 validation probe：Exact Match 整数计数、正确有向邻接边整数计数、负 validation loss。所有候选及完整验证完成后冻结模型，再评测 OOD。',
              'Base validation probe 与 SFT full validation 集合不同，不能把二者直接作为同集合提升；两种 OOD 结果的图片、排列、prompt、解码与分母完全相同。',
              '无效排列保留在分母，pairwise/adjacency/position计0、Kendall τ计-1。Satellite 五城 teacher 输出未用于 SFT 或 checkpoint 选择。',
              '未执行人工视觉审核；自动筛选不能保证所有 teacher 描述均有可靠视觉依据。不同后端和批处理数值可能改变 greedy 输出，结果仅按本次冻结协议比较。','',
              f'最佳 adapter：`{RUN}/training/checkpoint-{best["step"]}`。完整恢复状态保留于 checkpoint {retained}。',
              f'逐条预测：`{RUN}/vllm/evaluations/`；完整核验：`{RUN}/final_audit.json`；派生数据：`{DATA}`。',
              '全部模型来源、冻结源文件/派生输入哈希、训练范围、实际GPU归属、梯度和逐条评分已独立核验；CPU恢复检查与实际双卡25→26恢复的范围见各自核验文件。','']
    (RUN/'experiment_result.md').write_text('\n'.join(lines),encoding='utf-8')
    print('SATELLITE_FINAL_AUDIT_VERIFIED',best['step'],flush=True)


if __name__=='__main__': main()
