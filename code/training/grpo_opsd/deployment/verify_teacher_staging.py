"""CPU contract checks using the pinned native auto_await, without GPU workers."""
import asyncio
from contextlib import redirect_stdout
from inspect import unwrap
from io import StringIO
import json
from types import SimpleNamespace
from unittest.mock import patch

from staged_teacher import TeacherStager
from verl.utils.ray_utils import auto_await


class Replica:
    world_size = 1

    def __init__(self):
        self.awake = True
        self.calls = []

    async def sleep(self):
        self.calls.append('sleep');self.awake = False

    async def wake_up(self):
        self.calls.append('wake_up');self.awake = True


def fixture(fail=False, release=True):
    replicas = [Replica(), Replica()]
    output = SimpleNamespace(meta_info={'timing': {'agent_time': 7}}, responses=object())

    class Manager:
        @auto_await
        async def generate_sequences(self, prompts):
            assert all(replica.awake for replica in replicas)
            assert prompts.meta_info['global_steps'] in (1, 2)
            if fail:
                raise RuntimeError('test worker failure')
            return output

    def memory():
        return {str(gpu): {'used_mib': 70000 if replica.awake or not release else 45000,
                          'free_mib': 10000 if replica.awake or not release else 35000}
                for gpu, replica in zip((1, 2), replicas, strict=True)}

    return TeacherStager(Manager(), replicas, None, memory), replicas, output


def verify():
    staged, replicas, output = fixture()
    staged.initialize()
    assert not staged.awake and not any(replica.awake for replica in replicas)
    for step in (1, 2):
        prompts = SimpleNamespace(meta_info={'global_steps': step})
        result = staged.generate_sequences(prompts)
        assert result is output
        assert result.meta_info['timing']['agent_time'] == 7
        assert not staged.awake and not staged.active and not staged.fault
    assert all(replica.calls == ['sleep', 'wake_up', 'sleep', 'wake_up', 'sleep'] for replica in replicas)
    assert [event['operation'] for event in staged.events] == ['sleep', 'wake_up', 'sleep', 'wake_up', 'sleep']

    async def async_call():
        other, _, expected = fixture()
        await other.initialize()
        assert await other.generate_sequences(SimpleNamespace(meta_info={'global_steps': 1})) is expected
    asyncio.run(async_call())

    failed, replicas, _ = fixture(fail=True)
    failed.initialize()
    try:
        failed.generate_sequences(SimpleNamespace(meta_info={'global_steps': 1}))
    except RuntimeError as error:
        assert str(error) == 'test worker failure'
    else:
        raise AssertionError('Worker failure was swallowed')
    assert failed.fault and failed.awake and not failed.active
    assert all(replica.calls == ['sleep', 'wake_up'] for replica in replicas)
    try:
        failed.generate_sequences(SimpleNamespace(meta_info={'global_steps': 2}))
    except AssertionError:
        pass
    else:
        raise AssertionError('A failed rollout was allowed to resume')

    unreleased, _, _ = fixture(release=False)
    try:
        unreleased.initialize()
    except AssertionError:
        pass
    else:
        raise AssertionError('Ineffective sleep was accepted')
    assert unreleased.fault
    from run_deployment import native_run_with_staged_teacher
    import verl.trainer.main_ppo_v0 as native
    from staged_teacher import StagedTeacherTrainer
    original = unwrap(native.TaskRunner.__ray_metadata__.modified_class.run)
    staged_run = native_run_with_staged_teacher()
    assert staged_run.__code__ is original.__code__
    assert 'RayPPOTrainer' in staged_run.__code__.co_names
    assert not staged_run.__code__.co_freevars
    assert staged_run.__globals__['RayPPOTrainer'] is StagedTeacherTrainer
    assert all(staged_run.__globals__[key] is value for key, value in original.__globals__.items()
               if key != 'RayPPOTrainer')
    # Execute the pinned runner with CPU stand-ins. Checking only its globals
    # misses a Ray tracing closure that silently calls the unchanged trainer.
    calls = []
    class Trainer:
        def __init__(self, **kwargs):
            calls.append('staged_constructor')
        def init_workers(self):
            calls.append('init_workers')
        def fit(self):
            calls.append('fit')
    from omegaconf import OmegaConf
    config = OmegaConf.create({'actor_rollout_ref': {'model': {}},
                              'data': {'train_files': [], 'val_files': []}})
    noop = lambda *args, **kwargs: None
    runner = SimpleNamespace(add_actor_rollout_worker=lambda c: (object(), object()),
                             add_critic_worker=noop, add_reward_model_resource_pool=noop,
                             add_teacher_model_resource_pool=noop, add_ref_policy_worker=noop,
                             init_resource_pool_mgr=lambda c: object(), role_worker_mapping={})
    replacements = {'RayPPOTrainer': Trainer, 'validate_config': noop,
                    'need_reference_policy': lambda c: True, 'need_critic': lambda c: False,
                    'create_rl_dataset': lambda *args, **kwargs: object(),
                    'create_rl_sampler': lambda *args, **kwargs: object()}
    with patch.dict(staged_run.__globals__, replacements), \
            patch('verl.utils.config.omega_conf_to_dataclass',
                  return_value=SimpleNamespace(tokenizer=object(), processor=None)), \
            redirect_stdout(StringIO()):
        staged_run(runner, config)
    assert calls == ['staged_constructor', 'init_workers', 'fit'], calls
    assert original.__globals__['RayPPOTrainer'] is native.RayPPOTrainer
    return {'status': 'passed', 'scope': 'CPU lifecycle contract only; GPU release and scoring require native trial',
            'sync_and_async_calls': True, 'output_identity_preserved': True,
            'sleep_after_successful_gather_only': True, 'failure_blocks_next_update': True,
            'ineffective_memory_release_rejected': True, 'native_taskrunner_code_preserved': True,
            'unwrapped_constructor_executed': True, 'native_module_constructor_unchanged': True}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
