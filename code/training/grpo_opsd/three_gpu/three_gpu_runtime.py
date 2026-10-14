"""Three CPU request workers, each admitting at most 32 pending responses."""
import asyncio
import hashlib

import ray
from streetview_runtime import DATA, TEACHER_HASH, StreetViewWorker, StreetViewManager
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, AgentLoopManager


class ThreeGPUWorker(StreetViewWorker):
    def __init__(self, *args, **kwargs):
        # The original single-card admission check requires num_workers to divide
        # 16. Here each of three workers serves the entire three-replica manager.
        AgentLoopWorker.__init__(self, *args, **kwargs)
        self.template = (DATA / 'teacher_prompt.txt').read_text(encoding='utf-8')
        assert hashlib.sha256(self.template.encode()).hexdigest() == TEACHER_HASH
        r = self.config.actor_rollout_ref.rollout
        assert self.config.trainer.n_gpus_per_node == r.agent.num_workers == 3
        assert r.tensor_model_parallel_size == r.data_parallel_size == r.pipeline_model_parallel_size == 1
        self.pending = asyncio.Semaphore(2 * r.max_num_seqs)


class ThreeGPUManager(StreetViewManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(ThreeGPUWorker)
        AgentLoopManager.__init__(self, *args, **kwargs)
