"""
Lightweight shims for Wan2GP Cloud Client mode.

When running Wan2GP in Cloud Client mode (--mode cloud or --cloud), heavy local dependencies
like PyTorch CUDA, Triton, FlashAttention, and heavy ML frameworks are NOT required on the client machine.
This module installs a universal metapath finder and minimal stubs into sys.modules if they are not
already installed, allowing the Wan2GP Gradio UI, FastAPI endpoints, and cloud dispatchers to run with
only requirements-cloud.txt.
"""
from __future__ import annotations

from enum import Enum
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
import os
import sys
import types
from typing import Any, Callable, Dict, Optional


class _DummyContextManager:
    """No-op context manager and decorator for torch.no_grad, autocast, etc."""
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
    def __call__(self, fn_or_cls=None, *args, **kwargs):
        if fn_or_cls is None or not callable(fn_or_cls):
            return self
        return fn_or_cls


class _DummyDevice:
    """Mock torch.device."""
    def __init__(self, type_str="cpu", index=None):
        self.type = str(type_str).split(":")[0]
        self.index = index
    def __str__(self):
        return f"{self.type}:{self.index}" if self.index is not None else self.type
    def __repr__(self):
        return f"device(type='{self.type}')"


class _DummyTensor:
    """Minimal mock torch.Tensor for type checks and lightweight operations."""
    def __init__(self, data=None, dtype=None, device=None):
        self.data = data
        self.dtype = dtype or "float32"
        self.device = _DummyDevice("cpu")
        self.shape = (0,) if data is None else getattr(data, "shape", (0,))
    def cpu(self):
        return self
    def cuda(self, device=None):
        return self
    def to(self, *args, **kwargs):
        return self
    def detach(self):
        return self
    def clone(self):
        return self
    def numpy(self):
        import numpy as np
        return np.array(self.data) if self.data is not None else np.array([])
    def __repr__(self):
        return f"DummyTensor(shape={self.shape}, dtype={self.dtype})"


class MockClass:
    """Universal base class for dynamically mocked classes."""
    def __init__(self, *args, **kwargs):
        pass
    def __init_subclass__(cls, **kwargs):
        pass
    @classmethod
    def register(cls, *args, **kwargs):
        pass
    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls()
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class AutoConfigMock:
    """Mock transformers.AutoConfig supporting registration and lookup."""
    _registry: Dict[str, Any] = {}

    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs):
        if model_type in cls._registry:
            return cls._registry[model_type]
        raise ValueError(f"Unrecognized model type {model_type}")

    @classmethod
    def register(cls, model_type: str, config_class: Any, *args, **kwargs):
        cls._registry[model_type] = config_class


class ConfigMixin:
    """Mock diffusers.configuration_utils.ConfigMixin."""
    pass


class SchedulerMixin:
    """Mock diffusers.schedulers.scheduling_utils.SchedulerMixin."""
    pass


class KarrasDiffusionSchedulers(Enum):
    """Mock diffusers.schedulers.scheduling_utils.KarrasDiffusionSchedulers."""
    DPMSolverMultistep = 1
    Euler = 2
    EulerAncestral = 3


class MockModule(types.ModuleType):
    """Dynamic mock module that safely creates missing attributes and submodules."""
    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        if name == "AutoConfig":
            return AutoConfigMock
        if name == "init_empty_weights":
            return _DummyContextManager
        if name == "qtypes":
            return {}
        if name == "fix_text":
            return lambda text: text
        if name[0].isupper():
            cls = type(name, (MockClass,), {})
            setattr(self, name, cls)
            return cls
        val = MockModule(f"{self.__name__}.{name}")
        setattr(self, name, val)
        return val

    def __call__(self, *args, **kwargs) -> Any:
        # If decorating a class or callable, preserve it
        if args and (isinstance(args[0], type) or callable(args[0])):
            return args[0]
        return self


class MockLoader(Loader):
    """Loader for UniversalMockFinder."""
    def create_module(self, spec: ModuleSpec) -> types.ModuleType:
        mod = MockModule(spec.name)
        mod.__path__ = []
        mod.ConfigMixin = ConfigMixin
        mod.SchedulerMixin = SchedulerMixin
        mod.KarrasDiffusionSchedulers = KarrasDiffusionSchedulers
        mod.SchedulerOutput = object
        mod.register_to_config = lambda f: f
        mod.remove = lambda *a, **k: None
        mod.new_session = lambda *a, **k: None
        mod.TorchDispatchMode = object
        mod.Qwen3Config = type("Qwen3Config", (), {})
        mod.QModuleMixin = type("QModuleMixin", (), {})
        mod.register_qmodule = lambda *a, **k: (lambda f: f)
        mod.qtypes = {}
        mod.QEmbedding = type("QEmbedding", (), {})
        mod.QLinear = type("QLinear", (), {})
        mod.QConv2d = type("QConv2d", (), {})
        mod.QTensor = type("QTensor", (), {})
        mod.AutoConfig = AutoConfigMock
        mod.init_empty_weights = _DummyContextManager
        mod.fix_text = lambda text: text

        # Wire whisper LANGUAGES and TO_LANGUAGE_CODE
        whisper_langs = {
            "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
            "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
            "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
            "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
            "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
            "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
            "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
            "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
            "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
        }
        mod.LANGUAGES = whisper_langs
        mod.TO_LANGUAGE_CODE = {v: k for k, v in whisper_langs.items()}

        # Wire typing_extensions Unpack if available
        try:
            from typing_extensions import Unpack
            mod.Unpack = Unpack
        except ImportError:
            mod.Unpack = Any

        return mod

    def exec_module(self, module: types.ModuleType) -> None:
        pass


MOCK_HEAVY_PACKAGES = {
    "diffusers",
    "torchvision",
    "scipy",
    "torchaudio",
    "einops",
    "rembg",
    "ffmpeg",
    "whisper",
    "transformers",
    "optimum",
    "accelerate",
    "skimage",
    "smplfitter",
    "triton",
    "flash_attn",
    "onnxruntime",
    "matplotlib",
    "ftfy",
    "librosa",
    "av",
    "segment_anything",
}

KNOWN_SYSTEM_PACKAGES = {
    "os", "sys", "io", "re", "math", "time", "json", "random", "threading", "asyncio",
    "datetime", "pathlib", "typing", "typing_extensions", "dataclasses", "collections",
    "functools", "itertools", "traceback", "shutil", "tempfile", "copy", "hashlib",
    "importlib", "inspect", "subprocess", "weakref", "gc", "uuid", "base64", "urllib",
    "http", "socket", "warnings", "logging", "argparse", "enum", "contextlib", "struct",
    "platform", "glob",
}


class UniversalMockFinder(MetaPathFinder):
    """MetaPathFinder that intercepts heavy ML packages in cloud client mode."""
    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Optional[ModuleSpec]:
        top = fullname.split(".")[0]
        if top in KNOWN_SYSTEM_PACKAGES:
            return None
        if top in MOCK_HEAVY_PACKAGES:
            return ModuleSpec(fullname, MockLoader(), is_package=True)
        return None


def _create_mock_torch() -> types.ModuleType:
    """Build a comprehensive lightweight mock torch module for cloud client mode."""
    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = "2.6.0+cloud.client"
    torch_mod.__file__ = "/mock/torch/__init__.py"
    torch_mod.__path__ = ["/mock/torch"]
    torch_mod.Tensor = _DummyTensor
    torch_mod.Size = tuple
    torch_mod.device = _DummyDevice
    torch_mod.dtype = type
    torch_mod.Generator = type("Generator", (), {})

    # Data types
    torch_mod.float16 = "float16"
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.float32 = "float32"
    torch_mod.float64 = "float64"
    torch_mod.int8 = "int8"
    torch_mod.int16 = "int16"
    torch_mod.int32 = "int32"
    torch_mod.int64 = "int64"
    torch_mod.uint8 = "uint8"
    torch_mod.uint16 = "uint16"
    torch_mod.uint32 = "uint32"
    torch_mod.uint64 = "uint64"
    torch_mod.float8_e4m3fn = "float8_e4m3fn"
    torch_mod.float8_e5m2 = "float8_e5m2"
    torch_mod.bool = "bool"

    # Aliases
    torch_mod.float = torch_mod.float32
    torch_mod.double = torch_mod.float64
    torch_mod.half = torch_mod.float16
    torch_mod.int = torch_mod.int32
    torch_mod.long = torch_mod.int64
    torch_mod.short = torch_mod.int16
    torch_mod.complex64 = "complex64"
    torch_mod.complex128 = "complex128"

    # Tensor class aliases
    torch_mod.IntTensor = _DummyTensor
    torch_mod.FloatTensor = _DummyTensor
    torch_mod.LongTensor = _DummyTensor
    torch_mod.BoolTensor = _DummyTensor

    # Context managers
    torch_mod.no_grad = _DummyContextManager
    torch_mod.inference_mode = _DummyContextManager
    torch_mod.enable_grad = _DummyContextManager

    # CUDA submodule
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.is_available = lambda: False
    cuda_mod.device_count = lambda: 0
    cuda_mod.current_device = lambda: 0
    cuda_mod.get_device_name = lambda dev=None: "Cloud Worker"
    cuda_mod.get_device_capability = lambda dev=None: (0, 0)
    cuda_mod.empty_cache = lambda: None
    cuda_mod.synchronize = lambda dev=None: None
    cuda_mod.memory_allocated = lambda dev=None: 0
    cuda_mod.max_memory_allocated = lambda dev=None: 0
    cuda_mod.reset_peak_memory_stats = lambda dev=None: None
    cuda_mod.set_device = lambda dev=None: None
    cuda_mod.manual_seed = lambda s: None
    cuda_mod.manual_seed_all = lambda s: None

    amp_mod = types.ModuleType("torch.cuda.amp")
    amp_mod.autocast = _DummyContextManager
    cuda_mod.amp = amp_mod
    torch_mod.cuda = cuda_mod
    torch_mod.amp = amp_mod

    # Backends submodule
    backends_mod = types.ModuleType("torch.backends")
    mps_mod = types.ModuleType("torch.backends.mps")
    mps_mod.is_available = lambda: False
    mps_mod.is_built = lambda: False
    backends_mod.mps = mps_mod

    cuda_backends_mod = types.ModuleType("torch.backends.cuda")
    cuda_backends_mod.matmul = types.SimpleNamespace(allow_fp16_accumulation=True, allow_tf32=True)
    cuda_backends_mod.sdp_kernel = _DummyContextManager
    backends_mod.cuda = cuda_backends_mod

    cudnn_mod = types.ModuleType("torch.backends.cudnn")
    cudnn_mod.benchmark = False
    cudnn_mod.deterministic = False
    cudnn_mod.enabled = False
    backends_mod.cudnn = cudnn_mod
    torch_mod.backends = backends_mod

    # Version submodule
    version_mod = types.ModuleType("torch.version")
    version_mod.cuda = None
    version_mod.hip = None
    torch_mod.version = version_mod

    # _logging submodule
    logging_mod = types.ModuleType("torch._logging")
    logging_mod.set_logs = lambda **kwargs: None
    torch_mod._logging = logging_mod

    # compiler submodule
    compiler_mod = types.ModuleType("torch.compiler")
    compiler_mod.reset = lambda: None
    compiler_mod.is_compiling = lambda: False
    compiler_mod.disable = lambda fn=None, recursive=True: (lambda f: f) if fn is None else fn
    torch_mod.compiler = compiler_mod

    # jit submodule
    jit_mod = types.ModuleType("torch.jit")
    jit_mod.export = lambda f: f
    jit_mod.ignore = lambda f: f
    jit_mod.script = lambda f: f
    torch_mod.jit = jit_mod

    # ops submodule
    torch_mod.ops = MockModule("torch.ops")

    # NN and Functional submodules
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.__path__ = []
    class _NNModuleBase:
        def __init__(self, *args, **kwargs):
            pass
        def to(self, *args, **kwargs):
            return self
        def eval(self):
            return self
        def train(self, mode: bool = True):
            return self
        def __call__(self, *args, **kwargs):
            return None

    nn_mod.Module = _NNModuleBase
    nn_mod.Parameter = lambda x: x

    def _nn_getattr(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (MockClass,), {})
    nn_mod.__getattr__ = _nn_getattr

    functional_mod = types.ModuleType("torch.nn.functional")
    functional_mod.scaled_dot_product_attention = lambda *a, **k: None
    functional_mod.relu = lambda x, *a, **k: x
    functional_mod.gelu = lambda x, *a, **k: x
    functional_mod.silu = lambda x, *a, **k: x
    functional_mod.sigmoid = lambda x, *a, **k: x
    functional_mod.tanh = lambda x, *a, **k: x
    functional_mod.softmax = lambda x, *a, **k: x

    def _functional_getattr(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: a[0] if a else None
    functional_mod.__getattr__ = _functional_getattr
    nn_mod.functional = functional_mod

    init_mod = MockModule("torch.nn.init")
    nn_mod.init = init_mod

    nn_utils_mod = types.ModuleType("torch.nn.utils")
    nn_utils_mod.__path__ = []
    param_mod = types.ModuleType("torch.nn.utils.parametrizations")
    param_mod.weight_norm = lambda m, *a, **k: m
    nn_utils_mod.parametrizations = param_mod
    nn_utils_mod.weight_norm = lambda m, *a, **k: m
    nn_mod.utils = nn_utils_mod

    torch_mod.nn = nn_mod

    # library submodule (torch.library.custom_op)
    lib_mod = types.ModuleType("torch.library")
    def _dummy_custom_op(*a, **k):
        def decorator(fn):
            fn.register_fake = lambda *fa, **fk: (lambda f: f)
            return fn
        return decorator
    lib_mod.custom_op = _dummy_custom_op
    torch_mod.library = lib_mod

    # utils submodule (torch.utils._pytree, checkpoint, _python_dispatch)
    utils_mod = types.ModuleType("torch.utils")
    utils_mod.__path__ = []
    disp_mod = types.ModuleType("torch.utils._python_dispatch")
    disp_mod.TorchDispatchMode = object
    utils_mod._python_dispatch = disp_mod

    pytree_mod = types.ModuleType("torch.utils._pytree")
    pytree_mod.tree_flatten = lambda x: ([x], None)
    pytree_mod.tree_unflatten = lambda v, s: v[0]
    utils_mod._pytree = pytree_mod

    checkpoint_mod = types.ModuleType("torch.utils.checkpoint")
    checkpoint_mod.checkpoint = lambda f, *a, **k: f(*a, **k)
    utils_mod.checkpoint = checkpoint_mod

    model_zoo_mod = types.ModuleType("torch.utils.model_zoo")
    model_zoo_mod.load_url = lambda *a, **k: {}
    utils_mod.model_zoo = model_zoo_mod

    device_mod = types.ModuleType("torch.utils._device")
    device_mod._device_constructors = lambda: []
    utils_mod._device = device_mod

    torch_mod.utils = utils_mod

    # distributed submodule
    dist_mod = types.ModuleType("torch.distributed")
    dist_mod.is_initialized = lambda: False
    dist_mod.get_rank = lambda: 0
    dist_mod.get_world_size = lambda: 1
    dist_mod.ProcessGroup = type("ProcessGroup", (), {})
    torch_mod.distributed = dist_mod

    # overrides submodule
    overrides_mod = types.ModuleType("torch.overrides")
    class TorchFunctionMode:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    overrides_mod.TorchFunctionMode = TorchFunctionMode
    torch_mod.overrides = overrides_mod

    # Basic tensor creation stubs
    torch_mod.from_numpy = lambda arr: _DummyTensor(data=arr)
    torch_mod.zeros = lambda *a, **k: _DummyTensor()
    torch_mod.ones = lambda *a, **k: _DummyTensor()
    torch_mod.cat = lambda tensors, dim=0: _DummyTensor()
    torch_mod.tensor = lambda data, **k: _DummyTensor(data=data)
    torch_mod.save = lambda obj, f, **k: None
    torch_mod.load = lambda f, **k: {}
    torch_mod.manual_seed = lambda s: None

    return torch_mod


def _create_mock_mmgp() -> types.ModuleType:
    """Build a lightweight mock mmgp module for cloud client mode."""
    mmgp_mod = types.ModuleType("mmgp")
    mmgp_mod.__version__ = "3.8.2"

    class ProfileType:
        default = "default"
        fast = "fast"
        off = "off"
        def __getattr__(self, name):
            return name

    p_type = ProfileType()
    mmgp_mod.profile_type = p_type

    # offload
    offload_mod = types.ModuleType("mmgp.offload")
    offload_mod.PROFILE_DEFAULT = "default"
    offload_mod.PROFILE_FAST = "fast"
    offload_mod.PROFILE_OFF = "off"
    offload_mod.profile_type = p_type
    offload_mod.qtypes = {}
    offload_mod.QEmbedding = type("QEmbedding", (), {})
    offload_mod.QLinear = type("QLinear", (), {})
    offload_mod.QConv2d = type("QConv2d", (), {})
    offload_mod.offload_model = lambda *a, **k: None
    offload_mod.clean_memory = lambda *a, **k: None
    offload_mod.is_main_process = lambda: True
    offload_mod.get_cache = lambda *a, **k: None
    offload_mod.clear_caches = lambda *a, **k: None
    offload_mod.__getattr__ = lambda name: (lambda *a, **k: None)
    mmgp_mod.offload = offload_mod

    # safetensors2
    st2_mod = types.ModuleType("mmgp.safetensors2")
    st2_mod.safe_open = lambda *a, **k: None
    mmgp_mod.safetensors2 = st2_mod

    # quant_router
    qr_mod = types.ModuleType("mmgp.quant_router")
    qr_mod.unregister_handler = lambda *a, **k: None
    qr_mod.register_handler = lambda *a, **k: None
    qr_mod.register_file_extension = lambda *a, **k: None
    qr_mod.get_quantization_tokens = lambda *a, **k: []
    qr_mod.get_handler = lambda *a, **k: None
    qr_mod.get_handlers = lambda *a, **k: []
    qr_mod.__getattr__ = lambda name: (lambda *a, **k: [] if "token" in name or "list" in name else None)
    mmgp_mod.quant_router = qr_mod

    return mmgp_mod


def is_cloud_mode_active() -> bool:
    """Check if Cloud mode is currently requested or active."""
    if os.environ.get("WAN2GP_MODE", "").strip().lower() == "cloud":
        return True
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg in ("--cloud", "-cloud"):
            return True
        if arg.startswith("--mode=") and arg.split("=", 1)[1].strip().lower() == "cloud":
            return True
        if arg == "--mode" and i + 1 < len(argv) and argv[i + 1].strip().lower() == "cloud":
            return True
    return False


def install_cloud_shims(force: bool = False) -> bool:
    """
    Install lightweight shims and import interceptors into sys.modules.
    Returns True if shims were installed, False if local stack is preserved.
    """
    if not (force or is_cloud_mode_active()):
        return False

    # 1. Install universal MetaPathFinder for heavy ML packages
    if not any(isinstance(finder, UniversalMockFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, UniversalMockFinder())

    # 2. Torch shim (only if real torch is not installed or import fails)
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            torch_mock = _create_mock_torch()
            sys.modules["torch"] = torch_mock
            sys.modules["torch.cuda"] = torch_mock.cuda
            sys.modules["torch.cuda.amp"] = torch_mock.cuda.amp
            sys.modules["torch.amp"] = torch_mock.amp
            sys.modules["torch.backends"] = torch_mock.backends
            sys.modules["torch.backends.mps"] = torch_mock.backends.mps
            sys.modules["torch.backends.cuda"] = torch_mock.backends.cuda
            sys.modules["torch.backends.cudnn"] = torch_mock.backends.cudnn
            sys.modules["torch.version"] = torch_mock.version
            sys.modules["torch._logging"] = torch_mock._logging
            sys.modules["torch.compiler"] = torch_mock.compiler
            sys.modules["torch.jit"] = torch_mock.jit
            sys.modules["torch.ops"] = torch_mock.ops
            sys.modules["torch.nn"] = torch_mock.nn
            sys.modules["torch.nn.functional"] = torch_mock.nn.functional
            sys.modules["torch.nn.init"] = torch_mock.nn.init
            sys.modules["torch.nn.utils"] = torch_mock.nn.utils
            sys.modules["torch.nn.utils.parametrizations"] = torch_mock.nn.utils.parametrizations
            sys.modules["torch.library"] = torch_mock.library
            sys.modules["torch.utils"] = torch_mock.utils
            sys.modules["torch.utils._python_dispatch"] = torch_mock.utils._python_dispatch
            sys.modules["torch.utils._pytree"] = torch_mock.utils._pytree
            sys.modules["torch.utils.checkpoint"] = torch_mock.utils.checkpoint
            sys.modules["torch.utils.model_zoo"] = torch_mock.utils.model_zoo
            sys.modules["torch.distributed"] = torch_mock.distributed
            sys.modules["torch.utils._device"] = torch_mock.utils._device
            sys.modules["torch.overrides"] = torch_mock.overrides

    # Safetensors torch shim
    try:
        import safetensors
        st_torch = types.ModuleType("safetensors.torch")
        st_torch.load_file = lambda *a, **k: {}
        st_torch.save_file = lambda *a, **k: None
        st_torch.safe_open = lambda *a, **k: _DummyContextManager()
        safetensors.torch = st_torch
        sys.modules["safetensors.torch"] = st_torch
        if not hasattr(safetensors, "safe_open"):
            safetensors.safe_open = lambda *a, **k: _DummyContextManager()
    except Exception:
        pass

    # 3. MMGP shim (always use lightweight mock in cloud mode)
    mmgp_mock = _create_mock_mmgp()
    sys.modules["mmgp"] = mmgp_mock
    sys.modules["mmgp.offload"] = mmgp_mock.offload
    sys.modules["mmgp.safetensors2"] = mmgp_mock.safetensors2
    sys.modules["mmgp.quant_router"] = mmgp_mock.quant_router

    return True
