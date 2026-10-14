"""Apply observation hooks and verified compatibility fixes to pinned verl."""
from pathlib import Path
import subprocess

ROOT = Path(__import__('os').environ.get('LFM_ROOT', '/path/to/visual_trajectory'))
repo = ROOT / 'runtime/verl'
revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True, timeout=10).strip()
assert revision == '704a9f01bda8b9863c02c113547191bf167bc88f'
changes = {
    'verl/utils/fsdp_utils.py': [(
        '            "lora_" in n\n            for n, _ in submodule.named_parameters()\n',
        '            "lora_" in f"{clean_prefix}.{n}"\n            for n, _ in submodule.named_parameters()\n'), (
        '                n: p\n                for n, p in submodule.named_parameters()\n',
        '                f"{clean_prefix}.{n}": p\n                for n, p in submodule.named_parameters()\n'), (
        '                f"{clean_prefix}.{key}": (\n',
        '                key: (\n')],
    'verl/workers/utils/padding.py': [(
        '    position_ids_nested = torch.nested.as_nested_tensor(position_ids_list, layout=torch.jagged)\n',
        '    # Explicit offsets avoid ambiguous ragged-axis inference for equal-length VLM inputs.\n'
        '    position_ids_nested = torch.nested.nested_tensor_from_jagged(\n'
        '        torch.cat(position_ids_list, dim=-1), offsets=cu_seqlens,\n'
        '        jagged_dim=2 if position_ids.dim() == 3 else 1,\n'
        '    )\n')],
    'verl/trainer/distillation/losses.py': [(
        '        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),\n',
        '        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),\n'
        '        "distillation/sampled_estimator_mean": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].mean()),\n'
        '        "distillation/clamp_fraction": Metric(AggregationType.MEAN,\n'
        '            (distillation_losses[response_mask_bool].abs() > loss_config.loss_max_clamp).float().mean()\n'
        '            if loss_config.loss_max_clamp is not None else 0.0),\n')],
    'verl/workers/engine/fsdp/transformer_impl.py': [(
        '        self._build_model_optimizer()\n',
        '        self._build_model_optimizer()\n'
        '        if os.environ.get("STREETVIEW_AUDIT_DIR"):\n'
        '            from streetview_audit import model_initialized\n'
        '            model_initialized(self)\n')],
    'verl/workers/engine_workers.py': [(
        '        batch_size_per_dp = data.shape[0]\n',
        '        batch_size_per_dp = data.shape[0]\n'
        '        if os.environ.get("STREETVIEW_AUDIT_DIR"):\n'
        '            from streetview_audit import batch_started\n'
        '            batch_started(data)\n'), (
        '                    actor_output = self.train_batch(mini_batch_td)\n',
        '                    actor_output = self.train_batch(mini_batch_td)\n'
        '                if os.environ.get("STREETVIEW_AUDIT_DIR"):\n'
        '                    from streetview_audit import minibatch_finished\n'
        '                    minibatch_finished(self, mini_batch_td, actor_output, batch_idx)\n')],
}
for relative, replacements in changes.items():
    path = repo / relative
    text = path.read_text()
    for original, patched in replacements:
        if patched in text:
            continue
        assert text.count(original) == 1, relative
        text = text.replace(original, patched)
    compile(text, str(path), 'exec')
    path.write_text(text)
diff = subprocess.check_output(['git', '-C', str(repo), 'diff', '--', *changes], timeout=10)
(ROOT / 'runs/streetview_grpo_opsd_qwen3vl8b_lora_v1/provision/verl_audit.patch').write_bytes(diff)
print('Applied invariant hooks, position-ID ragged-axis fix and qualified LoRA leaf-name fix', flush=True)
