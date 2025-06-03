
import os

def is_distributed():
    """Returns True if running under torch.distributed context."""
    return int(os.environ.get('WORLD_SIZE', 1)) > 1

def get_world_size():
    return int(os.environ.get('WORLD_SIZE', 1))

def get_rank():
    return int(os.environ.get('RANK', 0))

def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))

def is_main_process():
    return get_rank() == 0