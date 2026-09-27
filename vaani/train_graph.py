"""CUDA-graph training step (low-delay plan Task 4b, `perf.numerics.cuda_graph`).

Forward, loss and backward of one run are captured into one CUDA graph with static input buffers. B32 and the crop
are fixed (20,000 items per epoch are 625 full batches: asserted by the caller), so every replay has the same shapes.
Before capture the step runs on a side stream a few times, so cuDNN's algorithm choice is fixed, and the gradient
tensors are allocated once and stay static (the optimizer zeroes them in place, never to None). The
over-parameterization products are formed inside the captured forward. BatchNorm running statistics update inside
the graph exactly as in eager mode.

The finiteness decision, clipping, the foreach optimizer, the scheduler and the EMA run eagerly after each replay,
so every training-control semantic is unchanged. A capture failure raises: it never falls back silently to eager.
Parity (tests/test_train_throughput.py, CUDA only): graphed and eager steps agree within 1e-6 relative over 50 steps
on identical weights and batches (losses, gradients, BatchNorm buffers, updated weights).
"""
from __future__ import annotations

import torch

WARMUP_ITERS = 3


class GraphedStep:
    def __init__(self, fwd, loss_fn, device, use_amp, low_delay, scored=None):
        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("perf.numerics.cuda_graph needs a CUDA device; set cuda_graph: false")
        self.fwd, self.loss_fn, self.device = fwd, loss_fn, torch.device(device)
        self.use_amp, self.low_delay, self.scored = use_amp, low_delay, scored
        self.graph = None
        self.static_in = self.static_target = self.static_clean = self.static_loss = None
        self.captured = "full"

    def _compute(self, inputs, target, is_clean):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            pred = self.fwd(*inputs)
        if self.low_delay:
            loss = self.loss_fn(pred.float(), target, None, is_clean, scored=self.scored)
        else:
            loss = self.loss_fn(pred.float(), target, None, is_clean)
        loss.backward()
        return loss

    @staticmethod
    def _clone(x):
        return None if x is None else x.detach().clone()

    def _copy_in(self, inputs, target, is_clean):
        for dst, src in zip(self.static_in, inputs):
            if dst is None:
                if src is not None:
                    raise RuntimeError("graphed step: an input that was None at capture is now set")
                continue
            if src.shape != dst.shape:
                raise RuntimeError(f"graphed step: input shape {tuple(src.shape)} differs from the captured "
                                   f"{tuple(dst.shape)} (fixed batch and crop are required)")
            dst.copy_(src, non_blocking=True)
        self.static_target.copy_(target, non_blocking=True)
        self.static_clean.copy_(is_clean, non_blocking=True)

    def capture(self, model, inputs, target, is_clean):
        self.static_in = [self._clone(x) for x in inputs]
        self.static_target, self.static_clean = self._clone(target), self._clone(is_clean)
        for p in model.parameters():                 # static gradient buffers
            if p.requires_grad and p.grad is None:
                p.grad = torch.zeros_like(p)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        bn_state = {k: v.detach().clone() for k, v in model.state_dict().items() if "running" in k or "num_batches" in k}
        with torch.cuda.stream(side):
            for _ in range(WARMUP_ITERS):            # fixes cuDNN's algorithm choice; gradients are zeroed afterwards
                self._compute(self.static_in, self.static_target, self.static_clean)
        torch.cuda.current_stream().wait_stream(side)
        with torch.no_grad():                        # warm-up must not move BatchNorm statistics or gradients
            sd = model.state_dict()
            for k, v in bn_state.items():
                sd[k].copy_(v)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
        self.graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(self.graph):
                self.static_loss = self._compute(self.static_in, self.static_target, self.static_clean)
        except Exception as e:
            raise RuntimeError(f"CUDA graph capture of the training step failed ({e!r}); fix the op or set "
                               "perf.numerics.cuda_graph: false (never a silent fallback)") from e

    def step(self, model, inputs, target, is_clean):
        """Replay forward + loss + backward; returns the static loss tensor. Gradients must have been zeroed in
        place (optimizer.zero_grad(set_to_none=False)) since the last step."""
        if self.graph is None:
            self.capture(model, inputs, target, is_clean)
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
        self._copy_in(inputs, target, is_clean)
        self.graph.replay()
        return self.static_loss
