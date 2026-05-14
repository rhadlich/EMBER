import os

import torch
import torch.distributed as dist


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank():
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    if torch.cuda.is_available():
        return dist.get_rank() % torch.cuda.device_count()
    return 0


def get_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_size():
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def init_process_group(method, batchnorm_group_size=1):
    if method == "nccl-openmpi":
        addrport = os.getenv("PMIX_SERVER_URI2").split("//")[1]
        address = addrport.split(":")[0]
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = "29500"
        rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 0))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    elif method == "nccl-slurm":
        rank = int(os.getenv("SLURM_PROCID"))
        world_size = int(os.getenv("SLURM_NTASKS"))
        address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    elif method == "nccl-slurm-pmi":
        rank = int(os.getenv("PMI_RANK"))
        world_size = int(os.getenv("SLURM_NTASKS"))
        address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    elif method == "nccl":
        dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            local_rank = int(os.getenv("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
    elif method == "dummy":
        pass
    elif method == "gloo":
        dist.init_process_group(backend="gloo")
    elif method == "mpi":
        dist.init_process_group(backend="mpi")
    else:
        raise NotImplementedError(f"Unsupported distributed init method: {method}")

    if dist.is_initialized():
        dist.barrier()

    num_groups = get_size() // batchnorm_group_size
    if num_groups * batchnorm_group_size != get_size():
        raise ValueError(
            "The number of ranks must be evenly divisible by batchnorm_group_size."
        )

    my_rank = get_rank()
    world_size = get_size()
    local_group = None
    if world_size > 1 and batchnorm_group_size > 1:
        for group_idx in range(num_groups):
            start = group_idx * batchnorm_group_size
            end = start + batchnorm_group_size
            ranks = list(range(start, end))
            tmp_group = torch.distributed.new_group(ranks=ranks)
            if my_rank in ranks:
                local_group = tmp_group
    return local_group
