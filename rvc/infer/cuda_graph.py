import gc
import threading
from collections import OrderedDict

import torch


MIB = 1024**2
GIB = 1024**3


def _tensor_signature(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.requires_grad,
    )


class _CapturedCall:
    def __init__(self, manager, function, inputs):
        self.inputs = tuple(torch.empty_like(value) for value in inputs)
        for static_input, input_tensor in zip(self.inputs, inputs):
            static_input.copy_(input_tensor)

        device = self.inputs[0].device
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        torch.cuda.empty_cache()
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        rng_state = torch.cuda.get_rng_state(device)

        try:
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(3):
                    warmup_output = function(*self.inputs)
            stream.synchronize()
            del warmup_output

            peak_growth = max(
                torch.cuda.max_memory_allocated(device) - baseline_allocated,
                torch.cuda.max_memory_reserved(device) - baseline_reserved,
            )
            torch.cuda.empty_cache()
            manager._prepare_capacity(device, peak_growth)
            reserved_before_capture = torch.cuda.memory_reserved(device)

            graph = torch.cuda.CUDAGraph()
            graph.register_generator_state(torch.cuda.default_generators[device_index])
            with torch.cuda.graph(graph), torch.no_grad():
                output = function(*self.inputs)
            torch.cuda.synchronize(device)
        finally:
            torch.cuda.set_rng_state(rng_state, device)

        reserved_growth = max(
            0, torch.cuda.memory_reserved(device) - reserved_before_capture
        )
        self.graph = graph
        self.output = output
        self.memory_bytes = max(peak_growth, reserved_growth)

    def replay(self, inputs):
        for static_input, input_tensor in zip(self.inputs, inputs):
            static_input.copy_(input_tensor, non_blocking=True)
        self.graph.replay()
        return self.output


class CUDAGraphManager:
    def __init__(self):
        self.enabled = False
        self.entries = OrderedDict()
        self.seen = OrderedDict()
        self.failures = OrderedDict()
        self.lock = threading.RLock()
        self.memory_bytes = 0

    def set_enabled(self, enabled):
        enabled = bool(enabled)
        if self.enabled == enabled:
            return
        self.enabled = enabled
        if not enabled:
            self.clear()

    def clear(self):
        with self.lock:
            devices = {
                entry.inputs[0].device for entry in self.entries.values()
            }
            for device in devices:
                try:
                    torch.cuda.synchronize(device)
                except Exception:
                    pass
            self.entries.clear()
            self.seen.clear()
            self.failures.clear()
            self.memory_bytes = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _limits(device):
        _, total_memory = torch.cuda.mem_get_info(device)
        if total_memory < 12 * GIB:
            max_entries = 1
        elif total_memory < 18 * GIB:
            max_entries = 2
        elif total_memory < 30 * GIB:
            max_entries = 4
        else:
            max_entries = 6
        memory_budget = int(total_memory * 0.25)
        free_margin = max(512 * MIB, int(total_memory * 0.10))
        return max_entries, memory_budget, free_margin

    def allows_stage(self, stage, device):
        if not self.enabled:
            return False
        _, total_memory = torch.cuda.mem_get_info(device)
        minimum_memory = {
            "generator": 0,
            "feature": 12 * GIB,
            "pitch": 18 * GIB,
        }
        return total_memory >= minimum_memory[stage]

    def _evict_one(self):
        _, entry = self.entries.popitem(last=False)
        self.memory_bytes = max(0, self.memory_bytes - entry.memory_bytes)
        del entry

    def _prepare_capacity(self, device, estimated_bytes):
        max_entries, memory_budget, free_margin = self._limits(device)
        if estimated_bytes > memory_budget:
            raise RuntimeError("estimated graph memory exceeds the global budget")

        while self.entries and (
            len(self.entries) >= max_entries
            or self.memory_bytes + estimated_bytes > memory_budget
        ):
            self._evict_one()

        gc.collect()
        torch.cuda.empty_cache()
        free_memory, _ = torch.cuda.mem_get_info(device)
        while self.entries and free_memory < estimated_bytes + free_margin:
            self._evict_one()
            gc.collect()
            torch.cuda.empty_cache()
            free_memory, _ = torch.cuda.mem_get_info(device)

        if free_memory < estimated_bytes + free_margin:
            free_mib = free_memory // MIB
            required_mib = (estimated_bytes + free_margin) // MIB
            raise RuntimeError(
                f"insufficient VRAM headroom ({free_mib} MiB free, "
                f"{required_mib} MiB required)"
            )

    @staticmethod
    def _trim_history(history):
        while len(history) > 64:
            history.popitem(last=False)

    def run(self, owner, namespace, function, *inputs):
        if (
            not self.enabled
            or not inputs
            or not all(torch.is_tensor(value) for value in inputs)
            or not inputs[0].is_cuda
        ):
            return function(*inputs)

        signature = (id(owner), str(namespace)) + tuple(
            _tensor_signature(value) for value in inputs
        )
        with self.lock:
            if signature in self.failures:
                self.failures.move_to_end(signature)
                return function(*inputs)

            entry = self.entries.get(signature)
            if entry is None:
                seen_count = self.seen.get(signature, 0) + 1
                self.seen[signature] = seen_count
                self.seen.move_to_end(signature)
                self._trim_history(self.seen)
                if seen_count < 2:
                    return function(*inputs)
                try:
                    entry = _CapturedCall(self, function, inputs)
                    self.entries[signature] = entry
                    self.memory_bytes += entry.memory_bytes
                except Exception as error:
                    self.failures[signature] = None
                    self._trim_history(self.failures)
                    try:
                        torch.cuda.synchronize(inputs[0].device)
                    except Exception:
                        pass
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(
                        f"CUDA Graph capture failed for {namespace}; "
                        f"using standard inference: {error}"
                    )
                    return function(*inputs)
            else:
                self.entries.move_to_end(signature)

            return entry.replay(inputs)
