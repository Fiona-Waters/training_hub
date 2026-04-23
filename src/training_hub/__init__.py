from .algorithms import Algorithm, Backend, AlgorithmRegistry, create_algorithm
from .algorithms.lora_grpo import lora_grpo, LoRAGRPOAlgorithm, ARTLoRAGRPOBackend
from .algorithms.lora_grpo_verl import VeRLLoRAGRPOBackend
from .algorithms.rewards import tool_call_reward, binary_reward
from .hub_core import welcome
from .visualization import plot_loss

__all__ = [
    'Algorithm',
    'Backend',
    'AlgorithmRegistry',
    'create_algorithm',
    'lora_grpo',
    'LoRAGRPOAlgorithm',
    'ARTLoRAGRPOBackend',
    'VeRLLoRAGRPOBackend',
    'tool_call_reward',
    'binary_reward',
    'welcome',
    'plot_loss',
]

try:
    from .algorithms.sft import sft, SFTAlgorithm, InstructLabTrainingSFTBackend
    __all__ += ['sft', 'SFTAlgorithm', 'InstructLabTrainingSFTBackend']
except ImportError:
    pass

try:
    from .algorithms.osft import OSFTAlgorithm, MiniTrainerOSFTBackend, osft
    __all__ += ['osft', 'OSFTAlgorithm', 'MiniTrainerOSFTBackend']
except ImportError:
    pass

try:
    from .algorithms.lora import lora_sft, LoRASFTAlgorithm, UnslothLoRABackend
    __all__ += ['lora_sft', 'LoRASFTAlgorithm', 'UnslothLoRABackend']
except ImportError:
    pass

try:
    from .profiling.memory_estimator import BasicEstimator, OSFTEstimatorExperimental, estimate, OSFTEstimator, LoRAEstimator, QLoRAEstimator
    __all__ += ['BasicEstimator', 'OSFTEstimatorExperimental', 'OSFTEstimator', 'LoRAEstimator', 'QLoRAEstimator', 'estimate']
except ImportError:
    pass
