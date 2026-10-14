"""Use native colocated teacher sleep only at complete-rollout boundaries."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import time

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.ray_utils import auto_await


def gpu_memory():
    output = subprocess.check_output(
        ['nvidia-smi', '-i', '1,2', '--query-gpu=index,memory.used,memory.free',
         '--format=csv,noheader,nounits'], text=True, timeout=5)
    result = {}
    for line in output.splitlines():
        index, used, free = [int(part.strip()) for part in line.split(',')]
        result[str(index)] = {'used_mib': used, 'free_mib': free}
    assert set(result) == {'1', '2'}
    return result


class TeacherStager:
    def __init__(self, manager, replicas, audit_dir, memory_probe=gpu_memory):
        assert len(replicas) == 2 and all(replica.world_size == 1 for replica in replicas)
        self.manager = manager
        self.replicas = replicas
        self.audit_dir = Path(audit_dir) if audit_dir is not None else None
        self.memory_probe = memory_probe
        self.awake = True
        self.active = False
        self.fault = False
        self.events = []

    def __getattr__(self, name):
        return getattr(self.manager, name)

    def record(self, event):
        event = {'time': time.time(), 'pid': os.getpid(), **event}
        self.events.append(event)
        if self.audit_dir is not None:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            with (self.audit_dir / 'teacher_staging.jsonl').open('a') as stream:
                stream.write(json.dumps(event) + '\n')

    async def transition(self, operation, rollout_step):
        assert not self.fault
        assert operation in ('wake_up', 'sleep')
        assert self.awake == (operation == 'sleep')
        before = self.memory_probe()
        started = time.monotonic()
        try:
            # Native vLLMReplica.sleep drains requests before level-1 sleep;
            # weights are backed up on CPU and restored by native wake_up.
            await asyncio.gather(*[getattr(replica, operation)() for replica in self.replicas])
            after = self.memory_probe()
            released = {gpu: before[gpu]['used_mib'] - after[gpu]['used_mib']
                        for gpu in ('1', '2')}
            if operation == 'sleep':
                assert min(released.values()) >= 8192, (
                    f'Teacher sleep did not release at least 8 GiB per GPU: {released}')
        except BaseException as error:
            self.fault = True
            self.record({'operation': operation + '_failed', 'rollout_step': rollout_step,
                         'error': type(error).__name__ + ': ' + str(error)})
            raise
        self.awake = operation == 'wake_up'
        elapsed = time.monotonic() - started
        self.record({'operation': operation, 'rollout_step': rollout_step,
                     'seconds': elapsed, 'teacher_awake': self.awake,
                     'gpu_before': before, 'gpu_after': after, 'released_mib': released})
        return elapsed

    @auto_await
    async def initialize(self):
        assert not self.active and self.awake
        await self.transition('sleep', 0)

    @auto_await
    async def generate_sequences(self, prompts):
        assert not self.active and not self.fault and not self.awake
        assert not prompts.meta_info.get('validate', False), 'Training has no generation validation'
        self.active = True
        step = int(prompts.meta_info['global_steps'])
        try:
            wake_seconds = await self.transition('wake_up', step)
            output = await self.manager.generate_sequences(prompts)
            # All agent outputs, including every teacher score, have returned.
            sleep_seconds = await self.transition('sleep', step)
        except BaseException as error:
            self.fault = True
            self.record({'operation': 'rollout_failed', 'rollout_step': step,
                         'teacher_awake': self.awake,
                         'error': type(error).__name__ + ': ' + str(error)})
            # Do not initiate sleep while other agents might still request a
            # teacher score after a failed gather. Native trainer aborts here.
            raise
        finally:
            self.active = False
        timing = output.meta_info.setdefault('timing', {})
        timing['teacher_stage_wake'] = wake_seconds
        timing['teacher_stage_sleep'] = sleep_seconds
        return output


class StagedTeacherTrainer(RayPPOTrainer):
    def init_workers(self):
        super().init_workers()
        assert self.use_teacher_policy and self.teacher_model_manager is not None
        managers = list(self.teacher_model_manager.teacher_model_managers.values())
        assert len(managers) == 1
        inference = managers[0].teacher_model_config.inference
        assert inference.free_cache_engine and inference.enable_sleep_mode
        replicas = managers[0].rollout_replicas
        staged = TeacherStager(self.async_rollout_manager, replicas,
                              Path(self.config.trainer.default_local_dir) / 'audit')
        staged.initialize()
        self.async_rollout_manager = staged
