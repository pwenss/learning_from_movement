"""Bound CPU request queues for native distributed rollout replicas."""
import asyncio
import ray
from streetview_runtime import StreetViewWorker, StreetViewManager
from verl.experimental.agent_loop.agent_loop import AgentLoopManager


class DeploymentWorker(StreetViewWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        r = self.config.actor_rollout_ref.rollout
        replicas = 2 // (r.tensor_model_parallel_size * r.data_parallel_size * r.pipeline_model_parallel_size)
        self.pending = asyncio.Semaphore(2 * r.max_num_seqs * replicas // r.agent.num_workers)


class DeploymentManager(StreetViewManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(DeploymentWorker)
        AgentLoopManager.__init__(self, *args, **kwargs)
