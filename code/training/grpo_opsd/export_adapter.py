"""Export the single-rank LoRA-only checkpoint using verl's native key conversion."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

import streetview_sft_common as common
from run_training import config_for
from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger


def export(checkpoint, output):
    checkpoint, output = Path(checkpoint), Path(output)
    assert json.loads((checkpoint / 'fsdp_config.json').read_text())['world_size'] == 1
    source = checkpoint / 'model_world_size_1_rank_0.pt'
    state = torch.load(source, map_location='cpu', weights_only=False)
    assert len(state) == 504 and all('lora_' in key and 'language_model.layers.' in key for key in state)
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in state.values())
    expected = {key.replace('.default.weight', '.weight'): value for key, value in state.items()}
    merger = FSDPModelMerger(ModelMergerConfig(
        operation='merge', backend='fsdp', local_dir=str(checkpoint), target_dir=str(output),
        hf_model_config_path=str(checkpoint / 'huggingface')))
    # Do not invoke full-model merge: this checkpoint deliberately omits frozen weights.
    merger.save_lora_adapter(state)
    assert not state
    adapter = output / 'lora_adapter'
    actual = load_file(adapter / 'adapter_model.safetensors')
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], value) for key, value in expected.items())
    config_file = adapter / 'adapter_config.json'
    config = json.loads(config_file.read_text())
    model = config_for('formal').actor_rollout_ref.model
    assert config['r'] == model.lora_rank and config['lora_alpha'] == model.lora_alpha
    # Preserve the exact language-only module selector rather than generic leaf names.
    config.update(target_modules=model.target_modules, base_model_name_or_path=model.path,
                  lora_dropout=0.0, inference_mode=True)
    common.atomic_json(config_file, config)
    common.atomic_json(adapter / 'export_verification.json', {
        'status': 'passed', 'checkpoint': str(checkpoint), 'tensor_count': len(actual),
        'source_sha256': common.hash_file(source),
        'adapter_sha256': common.hash_file(adapter / 'adapter_model.safetensors'),
        'base_model': model.path, 'dtype': 'float32', 'exact_tensor_equality': True})
    print(str(adapter), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output)
