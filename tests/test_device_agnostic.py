"""Device-agnostic behaviour of the XPU, CUDA and CPU plumbing.

Everything except the last class runs on any host (detection order, dispatch of the
synchronize and empty-cache helpers, autocast device type, input placement in
``predict_action`` and the inference worker). The last class compares a tiny
random-weight model on the CPU and on an XPU and is skipped without an XPU.

Run from the repository root: ``python -m pytest tests``.
"""

import queue
import sys
import unittest
from unittest import mock

import numpy as np
import torch
from transformers import BatchFeature

from tests.tiny_model import build_tiny_vitra, fixed_noise, tiny_inputs
from vitra.models.vla import vitra_paligemma
from vitra.utils import device as dev


def _patch_availability(*, xpu: bool, cuda: bool):
    """Make ``is_available`` report exactly the given backends."""

    def fake_is_available(device_type: str) -> bool:
        if device_type == "xpu":
            return xpu
        if device_type == "cuda":
            return cuda
        return device_type == "cpu"

    return mock.patch.object(dev, "is_available", side_effect=fake_is_available)


class DeviceDetectionTest(unittest.TestCase):
    def test_xpu_preferred_over_cuda(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")
            self.assertEqual(dev.available_device_types(), ["xpu", "cuda", "cpu"])

    def test_cuda_when_no_xpu(self):
        with _patch_availability(xpu=False, cuda=True):
            self.assertEqual(dev.default_device_type(), "cuda")

    def test_cpu_when_no_accelerator(self):
        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.default_device_type(), "cpu")
            self.assertEqual(dev.resolve_device(None), torch.device("cpu"))

    def test_default_device_carries_index_except_on_cpu(self):
        with _patch_availability(xpu=True, cuda=False):
            self.assertEqual(dev.default_device(1), torch.device("xpu:1"))
        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.default_device(1), torch.device("cpu"))

    def test_explicit_device_wins_over_detection(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda:1"))
            self.assertEqual(dev.resolve_device(torch.device("xpu:1")), torch.device("xpu:1"))

    def test_device_type_of_string_tensor_and_module(self):
        self.assertEqual(dev.device_type("xpu:0"), "xpu")
        self.assertEqual(dev.device_type(torch.device("cuda")), "cuda")
        self.assertEqual(dev.device_type(torch.zeros(1)), "cpu")
        self.assertEqual(dev.device_type(torch.nn.Linear(2, 2)), "cpu")

    def test_device_count_follows_detected_backend(self):
        with _patch_availability(xpu=True, cuda=False):
            with mock.patch.object(torch.xpu, "device_count", return_value=2):
                self.assertEqual(dev.device_count(), 2)
        with _patch_availability(xpu=False, cuda=True):
            with mock.patch.object(torch.cuda, "device_count", return_value=8):
                self.assertEqual(dev.device_count(), 8)
        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.device_count(), 0)

    def test_distributed_backend(self):
        self.assertEqual(dev.distributed_backend("xpu:0"), "xccl")
        self.assertEqual(dev.distributed_backend("cuda"), "nccl")
        self.assertEqual(dev.distributed_backend("cpu"), "gloo")
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.distributed_backend(), "xccl")


class DispatchTest(unittest.TestCase):
    """A wheel without CUDA raises on ``torch.cuda.synchronize``, so only the detected backend is touched."""

    def _assert_dispatch(self, func, method, *, xpu, cuda, expect, args=(), explicit=None):
        with _patch_availability(xpu=xpu, cuda=cuda):
            with (
                mock.patch.object(torch.xpu, method) as xpu_fn,
                mock.patch.object(torch.cuda, method) as cuda_fn,
            ):
                func(*args) if explicit is None else func(*args, explicit)
        calls = {"xpu": xpu_fn, "cuda": cuda_fn}
        for name, fn in calls.items():
            if name == expect:
                fn.assert_called_once()
            else:
                fn.assert_not_called()

    def test_synchronize(self):
        self._assert_dispatch(dev.synchronize, "synchronize", xpu=True, cuda=True, expect="xpu")
        self._assert_dispatch(dev.synchronize, "synchronize", xpu=False, cuda=True, expect="cuda")
        self._assert_dispatch(dev.synchronize, "synchronize", xpu=False, cuda=False, expect=None)
        self._assert_dispatch(dev.synchronize, "synchronize", xpu=True, cuda=True, expect="cuda", explicit="cuda:0")

    def test_empty_cache(self):
        self._assert_dispatch(dev.empty_cache, "empty_cache", xpu=True, cuda=True, expect="xpu")
        self._assert_dispatch(dev.empty_cache, "empty_cache", xpu=False, cuda=True, expect="cuda")
        self._assert_dispatch(dev.empty_cache, "empty_cache", xpu=False, cuda=False, expect=None)

    def test_set_device(self):
        self._assert_dispatch(dev.set_device, "set_device", xpu=True, cuda=True, expect="xpu", args=(1,))
        self._assert_dispatch(dev.set_device, "set_device", xpu=False, cuda=True, expect="cuda", args=(1,))
        self._assert_dispatch(dev.set_device, "set_device", xpu=False, cuda=False, expect=None, args=(1,))

    def test_current_device(self):
        with _patch_availability(xpu=True, cuda=True):
            with mock.patch.object(torch.xpu, "current_device", return_value=1):
                self.assertEqual(dev.current_device(), torch.device("xpu", 1))
        with _patch_availability(xpu=False, cuda=True):
            with mock.patch.object(torch.cuda, "current_device", return_value=3):
                self.assertEqual(dev.current_device(), torch.device("cuda", 3))
        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.current_device(), torch.device("cpu"))


class ModelDevicePlacementTest(unittest.TestCase):
    """The model used ``torch.autocast("cuda")`` and ``.to("cuda")``, which fail on a host without CUDA."""

    @classmethod
    def setUpClass(cls):
        cls.model = build_tiny_vitra()
        cls.image, cls.state, cls.state_mask, cls.fov = tiny_inputs()

    def _predict(self, **kwargs):
        with fixed_noise():
            return self.model.predict_action(
                self.image,
                "pick up the cup",
                self.state,
                self.state_mask,
                fov=self.fov,
                sample_times=2,
                num_ddim_steps=3,
                **kwargs,
            )

    def test_predict_action_runs_without_cuda(self):
        actions = self._predict()
        self.assertEqual(actions.shape, (2, 4, 192))
        self.assertTrue(np.isfinite(actions).all())
        self.assertTrue(np.abs(actions).max() > 0)

    def test_inputs_are_moved_to_the_model_device(self):
        moved_to = []
        real_to = BatchFeature.to

        def spy(self_, *args, **kwargs):
            moved_to.append(args[0] if args else kwargs.get("device"))
            return real_to(self_, *args, **kwargs)

        with mock.patch.object(BatchFeature, "to", spy):
            self._predict()
        self.assertEqual(moved_to, [torch.device("cpu")])

    def test_autocast_uses_the_model_device_type(self):
        seen = []
        real_autocast = torch.autocast

        def spy(device_type, *args, **kwargs):
            seen.append(device_type)
            return real_autocast(device_type, *args, **kwargs)

        with mock.patch.object(vitra_paligemma.torch, "autocast", spy):
            self._predict()
        # The model enters autocast for the image tower and for the language model.
        self.assertGreaterEqual(len(seen), 2)
        self.assertEqual(set(seen), {"cpu"})

    def test_cpu_autocast_bfloat16_runs(self):
        model = build_tiny_vitra(use_bf16=True)
        with fixed_noise():
            actions = model.predict_action(
                self.image, "x", self.state, self.state_mask, fov=self.fov, sample_times=1, num_ddim_steps=3
            )
        self.assertTrue(np.isfinite(actions).all())


class InferenceWorkerTest(unittest.TestCase):
    """The persistent VLA worker of ``scripts/inference_human_prediction.py`` honours ``--device``."""

    @classmethod
    def setUpClass(cls):
        # The script imports the visualization stack (PyTorch3D, MANO) at module level.
        # Only the VLA worker is under test, so those imports are stubbed.
        # (Stubs are added and removed key by key: restoring the whole of sys.modules would
        # also drop torch and transformers submodules imported in between.)
        cls._stubbed = [name for name in ("pytorch3d", "visualization.visualize_core") if name not in sys.modules]
        for name in cls._stubbed:
            sys.modules[name] = mock.MagicMock()
        sys.path.insert(0, "scripts")
        try:
            import inference_human_prediction as script
        except ImportError as error:  # a dependency of the script (decord, wandb, ...) is missing
            cls._remove_stubs()
            raise unittest.SkipTest(f"inference script is not importable here: {error}")
        finally:
            sys.path.remove("scripts")
        cls.script = script

    @classmethod
    def _remove_stubs(cls):
        for name in cls._stubbed:
            sys.modules.pop(name, None)

    @classmethod
    def tearDownClass(cls):
        cls._remove_stubs()
        sys.modules.pop("inference_human_prediction", None)

    def _run_worker(self, device):
        tasks, results = queue.Queue(), queue.Queue()
        tasks.put({"type": "shutdown"})
        model = mock.MagicMock()
        model.to.return_value = model
        with (
            mock.patch("vitra.models.load_model", return_value=model),
            mock.patch("vitra.utils.data_utils.load_normalizer"),
            mock.patch.object(self.script, "empty_cache") as empty_cache,
            mock.patch.object(self.script, "synchronize") as synchronize,
        ):
            self.script._vla_inference_worker({}, tasks, results, device)
        return model, results.get_nowait(), empty_cache, synchronize

    def test_model_moves_to_requested_device(self):
        model, message, empty_cache, synchronize = self._run_worker("xpu:1")
        model.to.assert_called_once_with(torch.device("xpu:1"))
        self.assertEqual(message, {"type": "ready"})
        empty_cache.assert_called_once_with(torch.device("xpu:1"))
        synchronize.assert_called_once_with(torch.device("xpu:1"))

    def test_default_device_is_detected(self):
        with _patch_availability(xpu=True, cuda=True):
            model, _, _, _ = self._run_worker(None)
        model.to.assert_called_once_with(torch.device("xpu"))


class TrainingEntryPointTest(unittest.TestCase):
    """Training used ``torch.cuda`` and the ``nccl`` backend directly."""

    def test_fsdp_uses_the_current_device_of_the_backend(self):
        with open("vitra/training/fsdp.py") as handle:
            source = handle.read()
        self.assertNotIn("torch.cuda", source)
        self.assertIn("device_id=current_device()", source)

    def test_train_script_has_no_direct_cuda_calls(self):
        with open("scripts/train.py") as handle:
            source = handle.read()
        self.assertNotIn("torch.cuda.set_device", source)
        self.assertNotIn("torch.cuda.empty_cache", source)
        self.assertNotIn('backend="nccl"', source)
        self.assertIn("distributed_backend()", source)


@unittest.skipUnless(dev.is_available("xpu"), "XPU not available")
class XpuParityTest(unittest.TestCase):
    """CPU versus XPU for the tiny random-weight model, same weights, same noise."""

    def _compare(self, use_bf16):
        model = build_tiny_vitra(use_bf16=use_bf16)
        image, state, state_mask, fov = tiny_inputs()

        def run():
            with fixed_noise():
                actions = model.predict_action(
                    image, "pick up the cup", state, state_mask, fov=fov, sample_times=2, num_ddim_steps=3
                )
            dev.synchronize()
            return actions

        cpu_actions = run()
        model.to("xpu")
        xpu_actions = run()
        self.assertGreater(np.abs(cpu_actions).max(), 1.0)
        return cpu_actions, xpu_actions

    def test_float32_actions_match_cpu(self):
        cpu_actions, xpu_actions = self._compare(use_bf16=False)
        mse = float(np.mean((cpu_actions - xpu_actions) ** 2))
        self.assertLess(mse, 1e-8, f"CPU versus XPU float32 MSE {mse:.3e}")
        np.testing.assert_allclose(xpu_actions, cpu_actions, atol=1e-3, rtol=1e-3)

    def test_bfloat16_autocast_actions_match_cpu(self):
        cpu_actions, xpu_actions = self._compare(use_bf16=True)
        mse = float(np.mean((cpu_actions - xpu_actions) ** 2))
        scale = float(np.mean(cpu_actions**2))
        self.assertLess(mse / scale, 5e-2, f"CPU versus XPU bfloat16 relative MSE {mse / scale:.3e}")


if __name__ == "__main__":
    unittest.main()
