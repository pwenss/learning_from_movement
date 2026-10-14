"""Independent vLLM generation audit, retaining the original full training audit."""
import json
import hashlib
import math
from pathlib import Path
import time

from streetview_sft_common import DATA, MODEL, ROOT, RUN, RUN_ID, atomic_json, hash_file, read_jsonl, score, summarize


METRICS = ['valid_answer', 'cot_format_valid', 'exact_match', 'truncated',
           'pairwise_accuracy', 'kendall_tau', 'directed_adjacency_accuracy', 'position_accuracy', 'output_tokens']


def check_summary(actual, expected):
    for group in ['overall', 'per_city', 'city_macro_average']:
        assert actual[group].keys() == expected[group].keys(), group
    groups = [(actual['overall'], expected['overall']),
              (actual['city_macro_average'], expected['city_macro_average'])]
    groups += [(actual['per_city'][c], expected['per_city'][c]) for c in expected['per_city']]
    for a, b in groups:
        for k, v in b.items():
            assert math.isfinite(a[k]) and math.isclose(a[k], v, rel_tol=1e-10, abs_tol=1e-10), (k, a[k], v)


def check_evaluation(directory, records, prompt):
    directory = Path(directory)
    rows = [row for rank in range(2) for row in read_jsonl(directory/f'predictions.rank{rank}.jsonl')]
    expected = {x['id']: x for x in records}
    assert len(rows) == len(expected) and {x['id'] for x in rows} == set(expected), directory
    for row in rows:
        source = expected[row['id']]
        assert all(row[k] == source[k] for k in ['city', 'shuffle', 'ground_truth']), row['id']
        assert type(row['output_tokens']) is int and 0 < row['output_tokens'] <= 3072
        recomputed = score(row['text'], source['ground_truth'], prompt['step_titles'], row['truncated'])
        assert all(row[k] == v for k, v in recomputed.items()), row['id']
    actual = json.loads((directory/'summary.json').read_text())
    check_summary(actual, summarize(rows))
    return actual


def verify_formal_resume(config):
    recovery = json.loads((RUN/'checkpoint100_recovery_verification.json').read_text())
    assert recovery['verified'] and recovery['global_step'] == 100
    assert recovery['optimizer_state_count'] == 504 and recovery['optimizer_steps'] == [100]
    assert recovery['scheduler_last_epoch'] == 100 and recovery['scheduler_step_count'] == 101
    assert recovery['adapter_tensor_count'] == 504 and len(recovery['rng']) == 2
    resume = json.loads((RUN/'epoch_validation_fix_restart.json').read_text())
    assert resume['status'] in ['verified_resume_active', 'verified_resume_completed']
    assert resume['cached_prefixes_verified'] is True
    assert resume['checkpoint_adapter_sha256'] == recovery['adapter_sha256']
    loaded = resume['resume_configuration']
    assert loaded['resume_from'] == str(RUN/'training/checkpoint-100')
    assert loaded['ddp_timeout_seconds'] == 345600
    assert loaded['code_hashes']['train_streetview_sft.py'] == resume['patched_train_sha256']
    keys = ['base_model', 'base_revision', 'attention', 'rank', 'alpha', 'dropout',
            'learning_rate', 'micro_batch', 'gradient_accumulation', 'world_size',
            'epochs', 'max_steps', 'seed', 'min_pixels', 'max_pixels', 'max_sequence',
            'image_size', 'images_per_trajectory', 'assistant_only_loss', 'target_field',
            'frozen_visual_encoder', 'target_modules', 'train_count', 'validation_loss_count']
    assert all(resume['original_configuration'][k] == loaded[k] == config[k] for k in keys)
    assert {x['rank'] for x in resume['cached_generation']} == {0, 1}
    for cached in resume['cached_generation']:
        path = RUN/'training/validation/checkpoint-100'/f'predictions.rank{cached["rank"]}.jsonl'
        lines = path.read_bytes().splitlines(keepends=True)
        assert len(lines) >= cached['completed']
        prefix = b''.join(lines[:cached['completed']])
        assert hashlib.sha256(prefix).hexdigest() == cached['sha256']
    for name, digest in config['code_hashes'].items():
        assert hash_file(ROOT/'code/training'/name) == digest
    deferred = RUN/'generation_defer_restart.json'
    if deferred.exists():
        change = json.loads(deferred.read_text())
        assert change['status'] in ['verified_resume_active','verified_resume_completed']
        assert change['original_code_verified'] is True
        assert change['global_step'] == 300 and change['optimizer_state_count'] == 504
        assert change['optimizer_steps'] == [300] and change['scheduler_last_epoch'] == 300
        assert change['scheduler_step_count'] == 301 and change['rng_ranks'] == [0,1]
        assert change['original_configuration']['code_hashes']['train_streetview_sft.py'] == resume['patched_train_sha256']
        assert config['resume_from'] == str(RUN/'training/checkpoint-300')
        assert config['generation_validation_schedule'] == 'after_training'
        assert all(change['original_configuration'][k] == config[k] for k in keys)
        for name, digest in change['new_code_hashes'].items():
            assert hash_file(ROOT/'code/training'/name) == digest
        for cached in change['cached_generation']:
            path = RUN/'training/validation/checkpoint-300'/f'predictions.rank{cached["rank"]}.jsonl'
            lines = path.read_bytes().splitlines(keepends=True)
            assert len(lines) >= cached['completed']
            assert hashlib.sha256(b''.join(lines[:cached['completed']])).hexdigest() == cached['sha256']
        callback = json.loads((RUN/'generation_defer_callback_verification.json').read_text())
        assert callback['verified'] and callback['eval_steps'] == [3,6]
        assert callback['saved_steps'] == [3,6] and callback['generation_calls'] == 0
        assert callback['resumed_generation_calls'] == 0
    workflow = json.loads((RUN/'epoch_checkpoint_validation_verification.json').read_text())
    assert workflow['verified'] is True
    formal = next(x for x in workflow['cases'] if not x['smoke'])
    assert formal['eval_steps'] == [3, 6]
    assert formal['saves'] == [{'step':3, 'fresh_eval':True}, {'step':6, 'fresh_eval':True}]
    assert all(not x['fresh_eval'] for case in workflow['cases'] if case['smoke'] for x in case['saves'])
    return {'checkpoint100_recovery':recovery, 'formal_resume':resume, 'callback_workflow':workflow}


def main():
    status = json.loads((RUN/'experiment_status.json').read_text())
    assert status['phase'] == 'complete', 'The experiment is still running or failed; no final result is certified'
    result = json.loads((RUN/'experiment_result.json').read_text())
    assert result['completed'] is True
    export = json.loads((DATA/'export_verification.json').read_text())
    assert export['verified'] is True
    counts = {'train':9999, 'validation_loss':932, 'validation_generation':1445, 'validation_probe':256, 'ood_test':3525}
    assert export['counts'] == counts
    records = {name: read_jsonl(DATA/f'{name}.jsonl') for name in counts}
    for name, n in counts.items():
        assert len(records[name]) == n
        assert hash_file(DATA/f'{name}.jsonl') == export['derived_hashes'][name], name
    train_ids = {x['id'] for x in records['train']}
    dev_ids = {x['id'] for x in records['validation_generation']}
    ood_ids = {x['id'] for x in records['ood_test']}
    assert not (train_ids & dev_ids or train_ids & ood_ids or dev_ids & ood_ids)
    cot = ROOT/'dataset/streetview/cot/train_qwen38_flash_cot_20260927'
    for name, digest in export['source_hashes'].items():
        assert hash_file(cot/name) == digest, 'Frozen source changed: '+name
    extra = json.loads((RUN/'additional_data_audit.json').read_text())
    assert extra['verified'] is True and extra['content_answers_match_ground_truth'] == 10931
    assert extra['ood_jpeg_headers_checked'] == 28200
    assert hash_file(DATA/'prompt.json') == extra['prompt_sha256']
    assert hash_file(ROOT/'dataset/streetview/split_manifest.json') == extra['base_split_sha256']
    training = RUN/'training'
    completed = json.loads((training/'training_complete.json').read_text())
    assert completed['completed'] is True and completed['step'] == 626 and completed['epoch'] >= 1.999
    config = json.loads((training/'configuration.json').read_text())
    assert config['smoke'] is False and config['max_steps'] == 626 and config['world_size'] == 2
    assert config['rank'] == 64 and config['micro_batch'] == 1 and config['gradient_accumulation'] == 16
    assert config['min_pixels'] == 65536 and config['max_pixels'] == 524288 and config['max_sequence'] == 8192
    parameters = json.loads((training/'trainable_parameters.json').read_text())
    assert len(parameters['matched_modules']) == 252 and sum(parameters['parameters'].values()) == 174587904
    assert all('language_model.layers.' in name and 'lora_' in name for name in parameters['parameters'])
    smoke = json.loads((RUN/'preflight/preflight_complete.json').read_text())
    assert smoke['passed'] is True and smoke['step'] == 26 and smoke['resume_from'].endswith('/checkpoint-25')
    resume_audit = json.loads((RUN/'resume_state_verification.json').read_text())
    assert resume_audit['verified'] is True and resume_audit['trainer_steps'] == [25,26]
    assert resume_audit['adapter_tensors_identical_after_zero_lr_resumed_update'] == 504
    formal_resume = verify_formal_resume(config)
    deferred_selection = json.loads((RUN/'deferred_checkpoint_selection.json').read_text())
    assert deferred_selection['verified'] and deferred_selection['training_completed_before_selection']
    assert deferred_selection['completed_steps'] == [100,200,300,313,400,500,600,626]
    assert config['generation_validation_schedule'] == 'after_training'
    memory = {}
    for rank in range(2):
        assert json.loads((RUN/f'gpu_preflight.rank{rank}.json').read_text())['passed'] is True
        placement = json.loads((training/f'gpu_placement.rank{rank}.json').read_text())
        expected_uuid = ['GPU-18cf292b-568d-aa6c-f0ac-3b7889f125aa','GPU-aded96b3-1317-039c-a3b5-cf8871defa07'][rank]
        assert placement['verified'] is True and placement['gpu_uuid'] == expected_uuid
        p = json.loads((training/f'progress.rank{rank}.json').read_text())
        assert p['step'] == 626 and p['gradient_checks'] > 0 and p['min_nonzero_adapter_tensors'] == 504
        memory[str(rank)] = p
    metrics = read_jsonl(training/'metrics.jsonl')
    assert all(math.isfinite(row[k]) for row in metrics for k in ['loss','eval_loss','grad_norm'] if k in row)
    best = json.loads((training/'best_checkpoint.json').read_text())
    checkpoint = Path(best['checkpoint'])
    assert checkpoint.resolve().parent == training.resolve() and checkpoint.is_dir()
    state = json.loads((checkpoint/'trainer_state.json').read_text())
    assert state['global_step'] == best['step']
    adapter_config = json.loads((checkpoint/'adapter_config.json').read_text())
    assert adapter_config['r'] == 64 and adapter_config['lora_alpha'] == 128
    assert adapter_config['lora_dropout'] == .05 and adapter_config['bias'] == 'none'
    assert adapter_config['target_modules'] == config['target_modules']
    frozen = json.loads((RUN/'frozen_final_checkpoint.json').read_text())
    assert frozen == best
    prompt = json.loads((DATA/'prompt.json').read_text())
    selected = None
    validation_results = []
    for step in [100,200,300,313,400,500,600,626]:
        summary = check_evaluation(RUN/'vllm/evaluations'/f'checkpoint-{step}', records['validation_probe'], prompt)
        losses = [row['eval_loss'] for row in metrics if row['step'] == step and 'eval_loss' in row]
        assert losses, step
        val_loss = losses[-1]
        key = [summary['overall']['exact_match'], summary['overall']['directed_adjacency_accuracy'], -val_loss]
        candidate = {'step':step, 'selection_key':key, 'summary':summary, 'validation_loss':val_loss}
        validation_results.append(candidate)
        if selected is None or key > selected['selection_key']:
            selected = candidate
    assert selected['step'] == best['step'] and selected['selection_key'] == best['selection_key']
    assert deferred_selection['selected_step'] == best['step']
    for step in [300,313,400,500,600,626]:
        assert json.loads((RUN/'vllm/evaluations'/f'checkpoint-{step}'/'stage.exit.json').read_text())['exit_code'] == 0
        assert (RUN/'vllm/evaluations'/f'checkpoint-{step}'/'summary.json').stat().st_mtime >= completed['time']
    evaluations = {}
    for name, manifest in [('baseline_validation_probe','validation_probe'), ('sft_validation_full','validation_generation'),
                           ('baseline_ood_full','ood_test'), ('sft_ood_full','ood_test')]:
        evaluations[name] = check_evaluation(RUN/'vllm/evaluations'/name, records[manifest], prompt)
        check_summary(result['evaluations'][name], evaluations[name])
    stages = ['gpu_preflight','export','preflight','preflight_resume','baseline_validation_probe','formal_training',
              'sft_validation_full','baseline_ood_full','sft_ood_full']
    for stage in stages:
        exit_path = (RUN/'vllm/evaluations'/stage/'stage.exit.json') if stage in evaluations else RUN/f'{stage}.exit.json'
        assert json.loads(exit_path.read_text())['exit_code'] == 0, stage
    protocol = RUN/'vllm/generation_protocol.json'
    digest = hash_file(protocol)
    engine_protocol = json.loads(protocol.read_text())
    assert engine_protocol['engine'] == 'vllm' and engine_protocol['version'] == '0.17.1'
    assert result['engine'] == best['engine'] == deferred_selection['engine'] == 'vllm'
    assert result['protocol_sha256'] == best['protocol_sha256'] == digest
    assert json.loads((RUN/'vllm/pilot_verified.json').read_text())['verified']
    for name, sha in engine_protocol['source_hashes'].items():
        assert hash_file(ROOT/'code/training'/name) == sha
    directories = [RUN/'vllm/evaluations'/f'checkpoint-{s}' for s in [100,200,300,313,400,500,600,626]]
    directories += [RUN/'vllm/evaluations'/n for n in evaluations]
    from streetview_sft_common import processor, render, text_ids
    tokenizer_processor = processor()
    expected_ids = text_ids(tokenizer_processor, render(tokenizer_processor, records['validation_probe'][0], prompt, False))
    expected_prompt_hash = hashlib.sha256(json.dumps(expected_ids).encode()).hexdigest()
    for directory in directories:
        metadata = json.loads((directory/'stage.exit.json').read_text())
        assert metadata['exit_code'] == 0 and metadata['started'] >= completed['time']
        assert metadata['protocol_sha256'] == digest
        assert hash_file(metadata['manifest']) == metadata['manifest_sha256']
        if metadata['adapter']:
            assert hash_file(Path(metadata['adapter'])/'adapter_model.safetensors') == metadata['adapter_sha256']
        for rank in range(2):
            for row in read_jsonl(directory/f'predictions.rank{rank}.jsonl'):
                assert row['engine'] == 'vllm-0.17.1' and row['protocol_sha256'] == digest
                assert row['adapter_sha256'] == metadata['adapter_sha256']
                assert row['prompt_token_sha256'] == expected_prompt_hash
                assert row['truncated'] == (row['output_tokens'] >= 3072 and row['last_token_id'] != 151645)
    transition = json.loads((RUN/'vllm_generation_transition.json').read_text())
    assert transition['status'] == 'transformers_stopped'
    for cache in transition['preserved_transformers_predictions']:
        assert hash_file(cache['path']) == cache['sha256']
    for rank in range(2):
        engine = json.loads((RUN/f'vllm/engine.rank{rank}.json').read_text())
        assert engine['uuid'] == ['GPU-18cf292b-568d-aa6c-f0ac-3b7889f125aa','GPU-aded96b3-1317-039c-a3b5-cf8871defa07'][rank]
        assert engine['vllm'] == '0.17.1' and engine['settings'] == engine_protocol['settings']
    verification = {'verified':True, 'generation_engine_protocol':engine_protocol, 'run_id':RUN_ID, 'time':time.time(), 'counts':counts,
                    'training':completed, 'best_checkpoint':str(checkpoint), 'best_step':best['step'],
                    'selection_recomputed':selected, 'validation_checkpoints':validation_results,
                    'source_and_split_hashes_unchanged':True, 'all_prediction_metrics_recomputed':True,
                    'adapter_sha256':hash_file(checkpoint/'adapter_model.safetensors'), 'memory':memory,
                    'manual_human_review':'not performed; frozen automated eligibility was used; see execution protocol',
                    'formal_resume_verification':formal_resume, 'evaluations':evaluations}
    atomic_json(RUN/'final_audit.json', verification)
    a = evaluations['baseline_ood_full']['overall']; b = evaluations['sft_ood_full']['overall']
    lines = ['# Street View Qwen3-VL-8B-Instruct 正式 SFT 结果','',
             f'实验 `{RUN_ID}` 已完成 2 个完整 epoch（626 次更新）。最佳 adapter 为 `{checkpoint}`。',
             f'同一冻结 3,525 条 OOD 轨迹：基础模型 Exact Match {a["exact_match"]:.2%}，SFT {b["exact_match"]:.2%}，变化 {(b["exact_match"]-a["exact_match"])*100:+.2f} 个百分点。','',
             '## 数据与训练条件','',
             '23 城 9,999 条训练样本；Toronto/Vienna 留出 932 条 loss validation、1,445 条生成 validation；五城 OOD 3,525 条。',
             '每条 8 张 1024×512 图片，冻结原始打乱排列和四步 prompt，assistant-only CoT 正文与答案监督。',
             'BF16、FlashAttention-2、双卡 DDP；语言层 LoRA r=64 / alpha=128 / dropout=0.05；有效 batch 32、学习率 5e-5、cosine、warmup 3%。视觉和全部基础权重冻结。',
             f'基础模型目录：`{MODEL}`；配置与环境版本：`training/configuration.json`、`environment.lock.txt`（相对于本 run）。','',
             '## 全量结果','',
             '| 集合 | N | Exact Match | Pairwise | Kendall τ | Directed adjacency | Position | 合法排列 | 四步格式 | 截断 | 平均输出 tokens |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    def table_row(label, v):
        return f'| {label} | {v["n"]} | {v["exact_match"]:.2%} | {v["pairwise_accuracy"]:.2%} | {v["kendall_tau"]:.4f} | {v["directed_adjacency_accuracy"]:.2%} | {v["position_accuracy"]:.2%} | {v["valid_answer"]:.2%} | {v["cot_format_valid"]:.2%} | {v["truncated"]:.2%} | {v["output_tokens"]:.1f} |'
    for name, summary in evaluations.items(): lines.append(table_row(name,summary['overall']))
    lines += ['', '基础模型 validation probe 为固定 256 条，SFT validation full 为 1,445 条；不能直接把这两个集合的分数作为同集合提升。OOD 基础模型与 SFT 使用完全相同的轨迹、排列、图片、prompt 和生成参数。','',
              '## 分城市结果','',
              '| 模型 / 城市 | N | Exact Match | Pairwise | Kendall τ | Directed adjacency | Position | 合法排列 | 四步格式 | 截断 | 平均输出 tokens |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in ['sft_validation_full','baseline_ood_full','sft_ood_full']:
        for city, values in evaluations[name]['per_city'].items(): lines.append(table_row(f'{name} / {city}',values))
        if name.endswith('ood_full'):
            lines.append(table_row(f'{name} / city macro',{'n':3525,**evaluations[name]['city_macro_average']}))
    lines += ['', '## 模型选择与核验','',
              '全部 8 个 checkpoint 使用固定 256 条 validation probe 自由生成；优先 Exact Match，再 directed adjacency，再 validation loss。最佳 checkpoint 的选择已独立重算核实；全量 validation 后冻结该 checkpoint，再运行 OOD。',
              '全部生成采用 greedy、max_new_tokens=3072、以 <|im_end|> 停止。无效排列保留在分母：pairwise、adjacency、position 计 0，Kendall τ 计 -1；完整原始输出保存在 vllm/evaluations/。',
              '派生 split 和原始 CoT/manifest/mapping 哈希在结束时再次核验；训练步数、梯度、LoRA 范围、断点恢复及全部预测指标通过 final_audit.json 的独立核验。',
              '正式训练分别从 checkpoint-100、checkpoint-300 接续：optimizer/scheduler/随机状态已核对，训练核心配置一致；两次接续前已完成的生成结果前缀 SHA-256 保持一致。按用户要求，第 300 步起的生成验证统一移至 626 步训练完成后进行；训练期间保留当前步 loss validation 和全部候选权重。epoch 与最后一步保存均取得当前步验证 loss。',
              f'生成后端为独立 vLLM 0.17.1 环境：两卡各一个单卡引擎、每卡并发 {engine_protocol["settings"]["max_num_seqs"]}，采用持续补充请求，BF16 base 与 LoRA、无量化，停止条件及生成上限不变。实际图片 processor grid 与完整 prompt token IDs 对齐；全部基础模型和八个候选用同一后端及并发设置重新生成，旧 Transformers 与旧并发设置的输出保留。不同后端、LoRA 推理精度和批处理可能改变 greedy 输出，不能把旧设置的分数混入本次选择。',
              'OOD 中 352 条轨迹曾用于 teacher 比较和选择，完整 3,525 条结果包含该子集；不能将全部 OOD 宣称为此前完全未使用的测试集。',
              '未执行人工视觉签核。执行协议使用冻结自动筛选样本，不能把自动核验报告解释为人工审查，也不能把 teacher 细节全都视为缩放后仍可见。','',
              '## 文件与复现','',
              f'- 最佳 adapter：`{checkpoint}`',
              f'- 完整核验：`{RUN}/final_audit.json`',
              f'- 逐轨迹输出及汇总：`{RUN}/vllm/evaluations/`',
              f'- 派生数据及原始来源哈希：`{DATA}`',
              '- 独立复核：在冻结环境中执行 code/training/audit_streetview_vllm_result.py；未完成的实验不会获得 verified 报告。','']
    (RUN/'experiment_result.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'verified':True,'best_checkpoint':str(checkpoint),'report':str(RUN/'experiment_result.md')},ensure_ascii=False))


if __name__ == '__main__':
    main()
