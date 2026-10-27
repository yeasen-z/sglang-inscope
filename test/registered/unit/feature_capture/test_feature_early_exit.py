"""Cutoff gating and real model-loop residual/boundary regression checks."""
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
import torch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
from sglang.srt.feature_capture.early_exit import resolve_layer
from sglang.srt.layers.layer_boundary.residual import batch as residual
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.models import qwen3_5 as model


def batch(**updates):
    value=NS(is_prefill_only=True,return_logprob=False,reqs=[NS(
        feature_capture=NS(layers=(16,32,48)),return_pooled_hidden_states=False,grammar=None)])
    value.__dict__.update(updates)
    return value


def test_gate(monkeypatch):
    monkeypatch.setenv('SGLANG_FEATURE_EARLY_EXIT_LAYER','48')
    assert resolve_layer(batch(),0,False)==48
    for update in [dict(is_prefill_only=False),dict(return_logprob=True),dict(reqs=[]),
                   dict(reqs=[NS(feature_capture=None,return_pooled_hidden_states=False,grammar=None)]),
                   dict(reqs=[NS(feature_capture=NS(layers=(64,)),return_pooled_hidden_states=False,grammar=None)]),
                   dict(reqs=[NS(feature_capture=NS(layers=(48,)),return_pooled_hidden_states=True,grammar=None)]),
                   dict(reqs=[NS(feature_capture=NS(layers=(48,)),return_pooled_hidden_states=False,grammar=object())])]:
        assert resolve_layer(batch(**update),0,False) is None
    assert resolve_layer(batch(),1,False) is None
    assert resolve_layer(batch(),0,True) is None
    monkeypatch.delenv('SGLANG_FEATURE_EARLY_EXIT_LAYER')
    assert resolve_layer(batch(),0,False) is None


@pytest.mark.parametrize('cutoff',[16,48,64,None])
def test_real_model_loop_captures_complete_residual_and_skips_tail(cutoff):
    visited=[]; captured={}
    class Recorder:
        def capture_hook(self,*,layer):
            if layer not in (0,16,32,48,64):return None
            def capture(value,**kw):captured[layer]=value.clone()
            return capture
    class Boundary:
        def capture_output(self,hidden_states,forward_batch):
            return residual.complete_output(hidden_states,forward_batch),residual.snapshot(hidden_states,forward_batch)
    class Layer:
        attn_boundary=Boundary()
        def __init__(self,index):self.index=index
        def __call__(self,*,hidden_states,forward_batch,captured_last_layer_outputs,**kw):
            if cutoff is not None and self.index>=cutoff:raise AssertionError('Executed discarded layer')
            visited.append(self.index)
            hidden=residual.fold(hidden_states,forward_batch)
            if captured_last_layer_outputs is not None:captured_last_layer_outputs.capture(hidden)
            return forward_batch.residual_stream.record(torch.ones_like(hidden),PLAIN_ADD)
    owner=NS(flashinfer_mnnvl_cutedsl_fusion=None,pp_group=NS(is_first_rank=True,is_last_rank=True,world_size=1),
             embed_tokens=lambda ids:torch.zeros((len(ids),2)),start_layer=0,end_layer=64,
             layers=[Layer(i) for i in range(64)],layers_to_capture=[],norm=None)
    forward=NS(feature_capture_early_exit_layer=cutoff,feature_capture_recorder=Recorder())
    def final_norm(hidden,batch,norm,*,capture,**kw):
        value=residual.snapshot(hidden,batch)
        if capture is not None:capture(value)
        residual.clear(batch)
        return value
    with patch.object(model,'get_global_expert_distribution_recorder',return_value=NS(with_current_layer=lambda i:nullcontext())),patch.object(residual,'final_norm',side_effect=final_norm):
        output=model.Qwen3_5ForCausalLM.forward(owner,torch.tensor([1]),torch.tensor([0]),forward)
    end=cutoff or 64
    assert visited==list(range(end))
    for layer,value in captured.items():
        assert torch.equal(value,torch.full((1,2),float(layer)))
    assert end in captured
    assert torch.equal(output,torch.full((1,2),float(end)))
    assert forward.residual_stream is None
