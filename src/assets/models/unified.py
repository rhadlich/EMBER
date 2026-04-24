import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from Loader import get_dataloader
import time
from pathlib import Path

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import destroy_process_group

import comm
from define_models import MLP, MLP_with_GRU_head, MSEWithDp
try:
    import hpo_general as hpo
except ImportError:
    hpo = None

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# $CONDA_PREFIX/bin/torchrun


# %%
def ddp_setup(method):
    comm_local_group = comm.init(method)
    return comm_local_group


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            val_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler,
            val_fn,
            train_method,
            distributed: bool,
            rank: int,
            world_size: int,
    ) -> None:
        self.global_rank = rank
        self.world_size = world_size
        self.distributed = distributed
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs_run = 0
        self.loss = float('inf')
        self.model = model
        self.val_fn = val_fn
        self.train_method = train_method

        self.loss_logger = float('inf')
        self.mse_logger = float('inf')
        self.mae_logger = float('inf')
        self.val_loss = float('inf')
        self.val_mse = float('inf')
        self.val_mae = float('inf')


        self.tf_count = 0

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        # source = source.float()
        # targets = targets.float()
        output = self.model(source)
        output = torch.squeeze(output)
        loss = criterion(output, targets)
        self.loss = loss
        self.loss_logger += loss
        self.mse_logger += mse(output, targets)
        self.mae_logger += mae(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_val(self,
                 loader,
                 ):
        self.model.eval()
        step_fn = self.val_fn or self.model
        val_logger = 0
        mse_logger = 0
        mae_logger = 0
        with torch.no_grad():
            for val_idx, (source, targets, filename, idx) in enumerate(loader):
                source = source.to(device)
                targets = targets.to(device)
                output = step_fn(source)
                # output = torch.squeeze(output)
                # targets = torch.squeeze(targets)
                loss = criterion(output, targets)
                mse_val = mse(output, targets)
                mae_val = mae(output, targets)
                val_logger += loss
                mse_logger += mse_val
                mae_logger += mae_val
        self.val_loss = val_logger / (val_idx + 1)
        self.val_mse = mse_logger / (val_idx + 1)
        self.val_mae = mae_logger / (val_idx + 1)
        self.model.train()

    def _run_epoch(self, epoch, *, tf_decay_exp=0.95):

        if self.distributed and hasattr(self.train_data.sampler, "set_epoch"):
            self.train_data.sampler.set_epoch(epoch)

        self.loss_logger = 0
        self.mse_logger = 0
        self.mae_logger = 0

        if self.train_method == 'default':
            for batch_idx, (source, targets, filename, idx) in enumerate(self.train_data):
                source = source.to(device)
                targets = targets.to(device)
                self._run_batch(source, targets)
        elif self.train_method == 'teacher_forcing':
            tf_ratio = 1 * tf_decay_exp ** epoch
            for batch_idx, (source, targets, filename, idx) in enumerate(self.train_data):
                source = source.to(device)

                # decide whether to use ground truth or autoregression
                if torch.rand(1) < tf_ratio:
                    p_in = targets
                else:
                    p_in = self.val_fn(source)
                input_tuple = (source, p_in)
                self._run_batch(input_tuple, targets)

        if self.global_rank == 0:
            print(f"[Epoch {epoch} | Loss: {(self.loss_logger / (batch_idx + 1)):.4f} | MSE: {self.mse_logger/(batch_idx+1)} | Steps: {len(self.train_data)}")

        self.avg_loss = self.loss_logger / (batch_idx + 1)
        self.avg_mae = self.mae_logger / (batch_idx + 1)
        self.avg_mse = self.mse_logger / (batch_idx + 1)

        self.scheduler.step()

    def train(self, max_epochs):
        # run training loop

        time_epoch = 0
        for epoch in range(self.epochs_run, max_epochs):
            tic_epoch = time.time()
            self._run_epoch(epoch)
            toc_epoch = time.time()
            time_epoch += toc_epoch - tic_epoch
            if self.global_rank == 0:
                print(f"Epoch time: {toc_epoch - tic_epoch}s")

        # average time per epoch and reduce across processes
        time_epoch /= max_epochs
        time_epoch = torch.tensor(time_epoch / self.world_size, device=device)
        if self.distributed:
            dist.all_reduce(time_epoch)
        if self.global_rank == 0:
            print(f'The average time per epoch is {time_epoch}s')
        self.time_epoch = time_epoch

        # evaluate model on validation set
        self._run_val(self.val_data)
        if self.distributed:
            dist.all_reduce(self.val_loss)
            self.val_loss /= self.world_size
        if self.global_rank == 0:
            print(f'The MSE of the model on the validation set is {self.val_loss}.')


def main(
    total_epochs,
    root_dir,
    node_type,
    method,
    num_layers,
    layer_exp,
    learning_rate,
    batch_size,
    dropout,
    n_trials,
    hpo_iters,
    distributed=False,
    output_dir="src/assets/models",
):
    # Initialize distributed process group (optional)
    if distributed:
        ddp_setup(method)

    # Get ranks and sizes
    rank = comm.get_rank() if distributed else 0
    size = comm.get_size() if distributed else 1

    # Get data loaders and data shape
    train_loader, train_size, validation_loader, validation_size = get_dataloader(
        root_dir,
        size,
        rank,
        batch_size,
        distributed=distributed,
    )
    data_sample, label_sample, _, _ = next(iter(train_loader))
    data_shape = data_sample[0].shape[0]
    label_shape = label_sample[0].shape[0]
    if rank == 0:
        print(f'Data Shape: {data_shape}')
        print(f'Labels Shape: {label_shape}')
    out_size = label_shape

    # Create model and wrap it with DDP
    torch.manual_seed(123)

    training_iters = max(1, hpo_iters)
    opt = None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for j in range(training_iters):

        # if hpo is desired, start hpo logger/sampler
        if hpo_iters:
            if hpo is None:
                raise ImportError("hpo_general is required when --hpo_iters > 0")
            if not opt:
                param_configs = {
                    "num_layers": {
                        "type": "int",
                        "low": 2,
                        "high": 10,
                    },
                    "layer_exp": {
                        "type": "int",
                        "low": 7,
                        "high": 10,
                    },
                    "dropout": {
                        "type": "int",
                        "low": 0,
                        "high": 3,
                    },
                    "learning_rate": {
                        "type": "float",
                        "low": 1e-6,
                        "high": 1e-3,
                        "scale": "log"
                    }
                }
                opt = hpo.HPOGeneral(
                    param_configs=param_configs,
                    metrics=['mse_dp', 'mse', 'mae'],
                )
            # sample
            sample = opt.sample()

            if rank == 0:
                print(f"Sampled parameters for iteration {j}: {sample}")

            # assign sampled parameters to variables
            num_layers = sample["num_layers"]
            layer_exp = sample["layer_exp"]
            dropout = sample["dropout"]*0.1
            learning_rate = sample["learning_rate"]

        model = MLP(
            input_dim=data_shape,
            output_dim=out_size,
            num_hidden=num_layers,
            hidden_exp=layer_exp,
            dropout=dropout,
        )
        val_fn = None
        train_method = 'default'
        # model = MLP_with_GRU_head(
        #     input_dim=data_shape,
        #     num_hidden_MLP=num_layers,
        #     hidden_exp_MLP=layer_exp,
        #     hidden_dim_GRU=64,
        #     seq_len=out_size
        # )
        # val_fn = model.inference_no_grad
        # train_method = 'teacher_forcing'

        model = model.to(device)
        # model = torch.compile(model, mode="max-autotune")
        if distributed:
            model = DDP(
                model,
                device_ids=None,
                output_device=None,
            )

        # save model weights
        filename_model = output_dir / f"model_weights_{node_type}.pth"
        if rank == 0:
            base_model = model.module if distributed else model
            torch.save(
                {
                    "model_state_dict": base_model.state_dict(),
                    "model_config": {
                        "input_dim": data_shape,
                        "output_dim": out_size,
                        "num_hidden": num_layers,
                        "hidden_exp": layer_exp,
                        "dropout": dropout,
                    },
                    "normalization": {
                        "expected_feature_order": ["inj_pressure", "inj_timing", "inj_duration"],
                    },
                },
                filename_model,
            )

        # Set optimizer, scheduler
        optimizer_lr = learning_rate * size if distributed else learning_rate
        optimizer = optim.AdamW(model.parameters(), lr=optimizer_lr)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)            # Not considering number of processes
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        times_total = np.zeros([n_trials])
        times_epoch = np.zeros([n_trials])
        loss_store = np.zeros([n_trials])
        mse_store = np.zeros([n_trials])
        mae_store = np.zeros([n_trials])
        start_times = []
        end_times = []

        # save optimizer weights
        filename_optimizer = output_dir / f"optimizer_weights_{node_type}.pth"
        if rank == 0:
            torch.save(optimizer.state_dict(), filename_optimizer)

        if distributed:
            dist.barrier()

        # loop for running the same model multiple times and record times
        for i in range(n_trials):

            # Start training model
            checkpoint = torch.load(filename_model, map_location=device)
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(torch.load(filename_optimizer, map_location=device))
            model.train()
            trainer = Trainer(
                model,
                train_loader,
                validation_loader,
                optimizer,
                scheduler,
                val_fn,
                train_method,
                distributed=distributed,
                rank=rank,
                world_size=size,
            )

            # run and time training
            start_time = time.strftime("%m/%d/%Y %H:%M:%S", time.localtime())
            tic = time.time()
            trainer.train(total_epochs)
            toc = time.time()
            end_time = time.strftime("%m/%d/%Y %H:%M:%S", time.localtime())
            t = toc - tic

            # compute metrics across processes
            t = torch.tensor(t, device=device)
            loss = trainer.val_loss
            mse_loss = trainer.val_mse
            mae_loss = trainer.val_mae
            if distributed:
                dist.all_reduce(t)
                dist.all_reduce(loss)
                dist.all_reduce(mse_loss)
                dist.all_reduce(mae_loss)
                t /= size
                loss /= size
                mse_loss /= size
                mae_loss /= size

            # store time and losses into np array
            times_total[i] = t.detach().cpu().numpy()
            times_epoch[i] = trainer.time_epoch.detach().cpu().numpy()
            loss_store[i] = loss.detach().cpu().numpy()
            mse_store[i] = mse_loss.detach().cpu().numpy()
            mae_store[i] = mae_loss.detach().cpu().numpy()
            start_times.append(start_time)
            end_times.append(end_time)

            # print things
            if rank == 0:
                print(
                    f'Iteration: {j} | Trial: {i} | Val Loss {loss} | Model Training Time: {toc - tic} seconds | Number of Threads: {torch.get_num_threads()} | World Size: {size}')
                base_model = model.module if distributed else model
                torch.save(
                    {
                        "model_state_dict": base_model.state_dict(),
                        "model_config": {
                            "input_dim": data_shape,
                            "output_dim": out_size,
                            "num_hidden": num_layers,
                            "hidden_exp": layer_exp,
                            "dropout": dropout,
                        },
                    },
                    filename_model,
                )

        if opt:
            # log performance metrics
            opt.log_performance(loss_store, metric="mse_dp")
            opt.log_performance(mse_store, metric="mse")
            opt.log_performance(mae_store, metric="mae")

        # save things in hdf5 file
        # if rank == 0:
        #     current_dir = os.getcwd()
        #     file_dir = os.path.join(current_dir, 'ampere_bm.h5')
        #     with h5py.File(file_dir, 'a') as f:
        #
        #         # check to see if file already existed/was populated
        #         if node_type not in f:
        #             # needs to create groups
        #             f.create_group(node_type)
        #
        #         # at this point groups for model size exist
        #         grp1 = f[node_type]
        #
        #         n_processes = str(size)
        #
        #         # check to see if a group for the current number of processes already exists, if it does then delete it
        #         if n_processes in grp1:
        #             del grp1[n_processes]
        #
        #         # create group for current number of processes
        #         grp2 = grp1.create_group(n_processes)
        #
        #         # create datasets under current group
        #         grp2.create_dataset(name='total time',
        #                             shape=(n_trials,),
        #                             dtype='f',
        #                             data=times_total, )
        #         grp2.create_dataset(name='epoch time',
        #                             shape=(n_trials,),
        #                             dtype='f',
        #                             data=times_epoch, )
        #         grp2.create_dataset(name='start time',
        #                             shape=(n_trials,),
        #                             dtype='S20',
        #                             data=start_times, )
        #         grp2.create_dataset(name='end time',
        #                             shape=(n_trials,),
        #                             dtype='S20',
        #                             data=end_times, )
        #         grp2.create_dataset(name='mse',
        #                             shape=(n_trials,),
        #                             dtype='f',
        #                             data=mse, )
        #         grp2.create_dataset(name='mae',
        #                             shape=(n_trials,),
        #                             dtype='f',
        #                             data=mae, )

    if opt:
        # save log
        opt.save_log('test.parquet')

    # end distributed process group
    if distributed and dist.is_initialized():
        destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train pressure predictor model (single process by default)')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    # parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('root_dir', type=str,
                        help='Root directory where train, validation, and test folders are located')
    parser.add_argument('node_type', type=str, help='Model tag used in checkpoint filename')
    parser.add_argument('--method', default='dummy', type=str, help='Distributed init method (dummy, gloo, nccl-*)')
    parser.add_argument('--num_layers', default=4, type=int, help='Number of hidden layers (default: 4)')
    parser.add_argument('--num_nodes_exp', default=10, type=int,
                        help='Exponential factor for 2**n nodes per layer (default: 8)')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning rate (default: 0.00064)')
    parser.add_argument('--batch_size', default=128, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--p', default=0.1, type=float, help='Dropout probability (default: 0.3)')
    parser.add_argument('--n_trials', default=1, type=int, help='Number of consecutive trials (default: 1)')
    parser.add_argument('--hpo_iters', default=0, type=int, help='Number of HPO samples (default: 0)')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--output_dir', default='src/assets/models', type=str, help='Directory to write checkpoints')
    args = parser.parse_args()

    # ---------------- HYPERPARAMETERS ----------------#
    input_size = 3  # number of features
    scheduler_step = 10
    scheduler_gamma = 0.5
    mse = nn.MSELoss(reduction='mean')
    criterion = MSEWithDp()
    mae = nn.L1Loss()

    main(
        args.total_epochs,
        args.root_dir,
        args.node_type,
        args.method,
        args.num_layers,
        args.num_nodes_exp,
        args.lr,
        args.batch_size,
        args.p,
        args.n_trials,
        args.hpo_iters,
        distributed=args.distributed,
        output_dir=args.output_dir,
    )

