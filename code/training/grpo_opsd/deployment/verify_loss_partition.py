"""CPU regression: native joint-loss gradients are invariant to DP splitting.

Unequal response masks exercise global-token weighting. This checks the loss
operators and micro-batch accumulation, not GPU/FSDP runtime correctness.
"""
import json
import torch
from omegaconf import OmegaConf
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss_vanilla, kl_penalty
from run_training import config_for


def verify():
    recipe = config_for('formal')
    actor = recipe.actor_rollout_ref.actor
    coefficient = recipe.distillation.distillation_loss.distillation_loss_coef
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    features = torch.arange(1, 17, dtype=torch.float64).reshape(4, 4) / 20
    old = -.5 - features
    reference = old - .1 * features
    teacher = old + .2 * features
    advantages = torch.tensor([1., -1., .4, -.6], dtype=torch.float64).unsqueeze(1).expand(4, 4)
    tokens = int(mask.sum())

    def loss(parameter, indices, world):
        config = OmegaConf.create({'clip_ratio': actor.clip_ratio, 'clip_ratio_low': actor.clip_ratio_low,
            'clip_ratio_high': actor.clip_ratio_high,
            'global_batch_info': {'dp_size': world, 'batch_num_tokens': tokens, 'global_batch_size': 4}})
        log_prob = old[indices] + parameter * features[indices]
        pg, _ = compute_policy_loss_vanilla(old[indices], log_prob, advantages[indices], mask[indices], config=config)
        opd_advantage = -(log_prob - teacher[indices]).clamp(-10, 10).detach()
        opd, _ = compute_policy_loss_vanilla(old[indices], log_prob, opd_advantage, mask[indices], config=config)
        reference_kl = agg_loss(kl_penalty(log_prob, reference[indices], actor.kl_loss_type),
                               mask[indices], actor.loss_agg_mode, **config.global_batch_info)
        return pg + coefficient * opd + actor.kl_loss_coef * reference_kl

    parameter = torch.tensor(.08, dtype=torch.float64, requires_grad=True)
    single = loss(parameter, [0, 1, 2, 3], 1)
    single.backward();expected_gradient = parameter.grad.item()
    result = {}
    for name, partitions in {'two_ranks': [[[0, 1]], [[2, 3]]],
                             'two_ranks_micro_one': [[[0], [1]], [[2], [3]]]}.items():
        gradients, values = [], []
        for batches in partitions:
            local = torch.tensor(.08, dtype=torch.float64, requires_grad=True)
            accumulated = sum(loss(local, batch, 2) for batch in batches)
            accumulated.backward();gradients.append(local.grad.item());values.append(accumulated.item())
        value, gradient = sum(values) / 2, sum(gradients) / 2
        assert abs(value - single.item()) < 1e-12
        assert abs(gradient - expected_gradient) < 1e-12
        result[name] = {'loss': value, 'gradient': gradient}
    return {'status': 'passed', 'scope': 'CPU native loss and gradient partition equivalence; not GPU runtime',
            'global_valid_tokens': tokens, 'per_rank_valid_tokens': [3, 7],
            'opsd_coefficient': coefficient, 'reference_kl_coefficient': actor.kl_loss_coef,
            'single': {'loss': single.item(), 'gradient': expected_gradient}, **result}


if __name__ == '__main__':
    torch.set_num_threads(4)
    print(json.dumps(verify(), indent=2))
