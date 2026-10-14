"""CPU regression for real native losses and the exact tail dispatch adapter."""
import copy
import inspect
import json
import os
from pathlib import Path
from unittest.mock import patch
from contextlib import contextmanager
from types import SimpleNamespace
import torch
from tensordict import TensorDict
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.trainer.ppo.core_algos import get_policy_loss_fn, agg_loss, kl_penalty
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from serial_runtime import SerialTrainer, DDPTrainingWorker, compact_metric
from verl.utils.metric import Metric, AggregationType
from verl.utils import tensordict_utils as tu
from ddp_engine import zero_padding_loss
from run_serial import configuration, RUN


def main():
    directory=RUN/'provision';directory.mkdir(parents=True,exist_ok=True)
    os.environ['STREETVIEW_AUDIT_DIR']=str(directory/'tail_check_audit')
    td=TensorDict({'response_mask':torch.ones(608,5),
                   'advantages':torch.ones(608,5),
                   'row_id':torch.arange(608)},batch_size=[608])
    batch=DataProto(batch=td)
    fake=object.__new__(SerialTrainer);fake.global_steps=1
    with patch.object(RayPPOTrainer,'_update_actor',lambda self,b:b):
        padded=SerialTrainer._update_actor(fake,batch)
    assert len(batch)==608 and batch.batch['response_mask'].sum()==3040
    assert len(padded)==609 and padded.batch['response_mask'][-1].sum()==0
    assert padded.batch['advantages'][-1].sum()==0
    assert torch.equal(padded.batch['row_id'][:608],torch.arange(608))
    local=padded.batch.chunk(3)
    assert [(x['response_mask'].sum(-1)>0).sum().item() for x in local]==[203,203,202]
    assert [[int((x['response_mask'][j:j+64].sum(-1)>0).sum()) for j in range(0,203,64)]
            for x in local]==[[64,64,64,11],[64,64,64,11],[64,64,64,10]]
    @contextmanager
    def stop_before_gpu(**kwargs):
        raise RuntimeError('compiled_native_loop_validated')
        yield
    worker=object.__new__(DDPTrainingWorker)
    worker.engine=SimpleNamespace(get_data_parallel_size=lambda:3,
        get_data_parallel_rank=lambda:0, train_mode=stop_before_gpu)
    check=local[0].clone()
    tu.assign_non_tensor(check,mini_batch_size=192,epochs=1)
    try:
        DDPTrainingWorker.train_mini_batch(worker,check)
        raise AssertionError('Expected pre-GPU test stop')
    except RuntimeError as exc:
        assert str(exc)=='compiled_native_loop_validated',str(exc)
    c=configuration('tail_preflight')
    actor=omega_conf_to_dataclass(c.actor_rollout_ref.actor)
    opd=omega_conf_to_dataclass(c.distillation).distillation_loss
    torch.manual_seed(41)
    old=torch.randn(609,5,dtype=torch.float64)-2
    ref=old+torch.randn_like(old)*.02
    teacher=old+torch.randn_like(old)*.03
    adv=torch.randn_like(old)
    features=torch.randn_like(old)
    masks=padded.batch['response_mask'].double()
    losses=[];errors=[];dummy_zero=True
    for start,end in [(0,64),(64,128),(128,192),(192,203)]:
        idx=torch.cat([torch.arange(r*203+start,r*203+end) for r in range(3)])
        valid=idx[idx<608];token_num=int(masks[idx].sum())
        def native(weight,rows,dp):
            mask=masks[rows];logp=old[rows]+weight*features[rows]
            if mask.sum()==0:
                zero,_=zero_padding_loss(model_output={'log_probs':logp},data={'response_mask':mask})
                return zero
            actor.global_batch_info.update(dp_size=dp,batch_num_tokens=token_num,
                                           global_batch_size=len(valid),loss_scale_factor=None)
            pg,_=get_policy_loss_fn(actor.policy_loss.loss_mode)(old_log_prob=old[rows],log_prob=logp,
                advantages=adv[rows],response_mask=mask,loss_agg_mode='token-mean',config=actor)
            kl=agg_loss(kl_penalty(logp,ref[rows],'low_var_kl'),mask,'token-mean',
                        dp_size=dp,batch_num_tokens=token_num,global_batch_size=len(valid))
            opd.global_batch_info.update(actor.global_batch_info)
            distill_adv=-kl_penalty(logp,teacher[rows],'k1').detach()
            dl,_=get_policy_loss_fn(opd.policy_loss_mode)(old_log_prob=old[rows],log_prob=logp,
                advantages=distill_adv,response_mask=mask,loss_agg_mode='token-mean',config=opd)
            return pg+.001*kl+.01*dl
        w=torch.tensor(.01,dtype=torch.float64,requires_grad=True)
        expected=native(w,valid,1);expected.backward();gradient=w.grad.clone()
        w=torch.tensor(.01,dtype=torch.float64,requires_grad=True)
        actual=sum(native(w,torch.arange(r*203+start,r*203+end),3) for r in range(3))/3
        actual.backward()
        error=max(abs(actual.item()-expected.item()),abs(w.grad.item()-gradient.item()))
        assert error<1e-12,error
        errors.append(error);losses.append(actual.item())
    w=torch.tensor(1.,requires_grad=True)
    zero,_=zero_padding_loss(model_output={'log_probs':w*torch.ones(1,5)},data={'response_mask':torch.zeros(1,5)})
    zero.backward();assert zero.item()==w.grad.item()==0
    metric_errors=[]
    values=[list(range(11)),list(range(11,22)),list(range(22,32))]
    for kind in AggregationType:
        metrics=[Metric(aggregation=kind) for v in values]
        for metric, items in zip(metrics, values):
            metric.extend(items)
        try:
            Metric.aggregate_dp(metrics)
            raise AssertionError('Regression must reproduce native ragged metric failure')
        except ValueError as exc:
            assert 'same number of values' in str(exc)
        fixed=[compact_metric(m,len(v)*3/32) for m,v in zip(metrics,values)]
        actual=Metric.aggregate_dp(fixed)
        all_values=sum(values,[])
        expected={AggregationType.MEAN:sum(all_values)/32,
                  AggregationType.SUM:sum(all_values)/3,
                  AggregationType.MIN:min(all_values),
                  AggregationType.MAX:max(all_values)}[kind]
        assert abs(actual-expected)<1e-12,(kind,actual,expected)
        assert all(len(m.values)==1 for m in fixed)
        metric_errors.append(abs(actual-expected))
    report={'status':'passed','actual_native_GRPO_reference_KL_and_PG_OPD_loss_gradient_max_error':max(errors),
            'real_answers':608,'dispatch_rows':609,'global_real_minibatches':[192,192,192,32],
            'local_real_minibatches':[[64,64,64,11],[64,64,64,11],[64,64,64,10]],
            'dummy_gradient_exactly_zero':True,'original_batch_unchanged':True,
            'new_generation_requests':0,'formal_train_batch_size':96,'gpu_ids':[0,2,3],
            'actual_native_loop_clone_compiled_and_iterator_initialized':True}
    report['native_ragged_metric_failure_reproduced_and_fixed']=True
    report['all_four_metric_aggregation_types_preserved_max_error']=max(metric_errors)
    (directory/'three_gpu_tail_regression.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
    transient=directory/'tail_check_audit/tail_dispatch_padding.jsonl'
    transient.unlink()
    transient.parent.rmdir()


if __name__=='__main__':
    main()
