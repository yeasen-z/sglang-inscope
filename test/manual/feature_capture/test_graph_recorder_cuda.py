"""Actual CUDA replay must update native feature positions and own its output."""

import unittest

import torch

from sglang.srt.feature_capture.graph import GraphFeatureCaptureRecorder
from sglang.srt.feature_capture.plan import SlotPlan


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGraphRecorderCUDA(unittest.TestCase):
    def test_changed_rows_and_previous_output_survive_replay(self):
        source = (
            torch.arange(32 * 64, device="cuda", dtype=torch.float32)
            .reshape(32, 64)
            .to(torch.bfloat16)
        )
        recorder = GraphFeatureCaptureRecorder(
            layers=(8, 16), capacity=32, device=source.device
        )
        hooks = [recorder.capture_hook(layer=layer) for layer in (8, 16)]
        first = (
            SlotPlan(layer=8, rows=(1, 7), segments=(("a", (1, 7)),), out_start=0),
            SlotPlan(layer=16, rows=(4,), segments=(("b", (4,)),), out_start=2),
        )
        recorder.prepare(first)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                for hook in hooks:
                    hook(source)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for hook in hooks:
                hook(source)
        graph.replay()
        previous = recorder.device_output()
        expected_previous = torch.cat((source[[1, 7]], source[[4]]))
        self.assertTrue(torch.equal(previous.tensor, expected_previous))
        second = (
            SlotPlan(layer=16, rows=(0, 31), segments=(("c", (0, 31)),), out_start=0),
        )
        source.add_(100)
        recorder.prepare(second)
        graph.replay()
        latest = recorder.device_output()
        self.assertEqual(latest.plan, second)
        self.assertTrue(torch.equal(latest.tensor, source[[0, 31]]))
        self.assertTrue(torch.equal(previous.tensor, expected_previous))
        self.assertNotEqual(previous.tensor.data_ptr(), latest.tensor.data_ptr())
        recorder.prepare(())
        graph.replay()
        self.assertIsNone(recorder.device_output())
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
