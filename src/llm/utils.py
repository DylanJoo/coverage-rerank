import torch
import json
import re
import os
import string
import time
import gc

def cleanup_vllm(llm):
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()

    del llm.model.llm_engine.model_executor.driver_worker
    del llm.model
    del llm

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    return 'vllm engine has been cleaned up'

def check_if_ampere():
    if not torch.cuda.is_available():
        return "CUDA is not available"
    
    # Get device properties
    device_props = torch.cuda.get_device_properties(0)  # For the first GPU
    
    # Get the GPU name and compute capability
    gpu_name = device_props.name
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    # Ampere GPUs have compute capability 8.x
    is_ampere = device_props.major == 8
    
    return is_ampere
