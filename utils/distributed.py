import os
import pickle
from typing import Any, Dict

import torch
import torch.distributed as dist


def init_distributed() -> bool:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank % torch.cuda.device_count()) if torch.cuda.is_available() else None
        setup_for_distributed(rank == 0)
        return True
    return False


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return (not is_distributed()) or dist.get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def setup_for_distributed(is_master: bool) -> None:
    """Disable printing when not in master process."""

    import builtins as __builtin__

    builtins_print = __builtin__.print

    def print(*args, **kwargs):  # type: ignore
        force = kwargs.pop("force", False)
        if is_master or force:
            builtins_print(*args, **kwargs)

    __builtin__.print = print  # type: ignore


def all_gather_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Gather dictionaries across ranks."""
    if not is_distributed():
        return data
    tensor = torch.tensor(bytearray(pickle.dumps(data)), dtype=torch.uint8, device="cuda")
    size = torch.tensor([tensor.numel()], device=tensor.device)
    sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, size)
    max_size = torch.stack(sizes).max()
    padded = torch.zeros(max_size, dtype=torch.uint8, device=tensor.device)
    padded[: tensor.numel()] = tensor
    gathered = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)
    output = {}
    for chunk, chunk_size in zip(gathered, sizes):
        bytes_list = chunk[: chunk_size.item()].cpu().numpy().tobytes()
        output.update(pickle.loads(bytes_list))
    return output
