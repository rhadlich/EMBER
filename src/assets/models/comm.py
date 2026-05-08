import os
import torch
import torch.distributed as dist


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    return rank


def get_local_rank():
    """
    Gets node local rank or returns zero if distributed is not initialized.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return 0

    # number of GPUs per node
    if torch.cuda.is_available():
        local_rank = dist.get_rank() % torch.cuda.device_count()
    else:
        local_rank = 0

    return local_rank


def get_size():
    """
    Gets size of communicator
    """
    if dist.is_available() and dist.is_initialized():
        size = dist.get_world_size()
    else:
        size = 1
    return size


def get_local_size():
    """
    Gets size of shared memory group
    """
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    if torch.cuda.is_available():
        local_size = torch.cuda.device_count()
    else:
        local_size = 1

    return local_size


def init(method, batchnorm_group_size=1):
    # get master address and port
    if method == "nccl-openmpi":
        addrport = os.getenv("PMIX_SERVER_URI2").split("//")[1]
        # use that URI
        address = addrport.split(":")[0]
        # use the default pytorch port
        port = "29500"
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = port
        rank = int(os.getenv('OMPI_COMM_WORLD_RANK', 0))
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 0))

        # init DDP
        dist.init_process_group(backend="nccl",
                                rank=rank,
                                world_size=world_size)

    elif method == "nccl-slurm":
        rank = int(os.getenv("SLURM_PROCID"))
        world_size = int(os.getenv("SLURM_NTASKS"))
        address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
        port = "29500"
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = port

        # init DDP
        dist.init_process_group(backend="nccl",
                                rank=rank,
                                world_size=world_size)

    elif method == "nccl-slurm-pmi":
        rank = int(os.getenv("PMI_RANK"))
        world_size = int(os.getenv("SLURM_NTASKS"))
        address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
        port = "29500"
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = port

        # init DDP
        dist.init_process_group(backend="nccl",
                                rank=rank,
                                world_size=world_size)
    elif method == "nccl":
        # Torchrun-compatible NCCL initialization via environment variables
        # (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, LOCAL_RANK).
        dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            local_rank = int(os.getenv("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
    elif method == "dummy":
        rank = 0
        world_size = 1
        pass
    elif method == "gloo":
        dist.init_process_group(backend="gloo")

    elif method == "mpi":
        # init DDP
        dist.init_process_group(backend="mpi")

    else:
        raise NotImplementedError()

    # make sure to call a barrier here in order for sharp to use the default comm:
    if dist.is_initialized():
        dist.barrier()

    # create local group
    num_groups = get_size() // batchnorm_group_size
    assert (
                num_groups * batchnorm_group_size == get_size()), "Error, the number of ranks have to be evenly divisible by batchnorm group size"
    my_rank = get_rank()
    world_size = get_size()
    local_group = None
    if world_size > 1 and batchnorm_group_size > 1:
        for i in range(num_groups):
            start = i * batchnorm_group_size
            end = start + batchnorm_group_size
            ranks = list(range(start, end))
            tmp_group = torch.distributed.new_group(ranks=ranks)
            if my_rank in ranks:
                local_group = tmp_group

    return local_group