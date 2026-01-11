import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import pandas as pd

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
device = th.device("cuda" if th.cuda.is_available() else "cpu")

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        flow_matching=False, # <--- [NEW] Flag to enable FM mode
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.flow_matching = flow_matching # <--- [NEW] Store the flag

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size 

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        
        self.ema_params = [
            copy.deepcopy(self.mp_trainer.master_params)
            for _ in range(len(self.ema_rate))
        ]

        self.use_ddp = False
        self.ddp_model = self.model
            
        self.train_log = pd.DataFrame(columns=['epoch', 'loss'])
        self.train_loss = []

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            modeldir = th.load('model.pt') # Adjust path as needed
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(modeldir)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            batch = batch.to(device)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
            self.train_loss.append(self.ls)
            self.train_log = self.train_log.append({'epoch': self.step, 'loss': self.ls}, ignore_index=True)
            
            if self.step % 100 == 0:
                self.train_log.to_csv('train_log.csv', index=False)
        
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(device)
            micro_cond = {
                k: v[i : i + self.microbatch].to(device)
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            
            # --- [NEW] Logic to switch between Diffusion and Flow Matching ---
            if self.flow_matching:
                # Flow Matching:
                # We skip the schedule_sampler (which gives discrete steps 0..1000).
                # The loss function handles continuous time sampling internally.
                # We create dummy vars for logging compatibility.
                t = th.zeros(micro.shape[0], device=device) 
                weights = th.ones(micro.shape[0], device=device)

                compute_losses = functools.partial(
                    self.diffusion.training_losses_flow_matching, # Call the FM function
                    self.ddp_model,
                    micro,
                    model_kwargs=micro_cond,
                )
            else:
                # Standard Diffusion:
                # Sample discrete timesteps using the schedule sampler
                t, weights = self.schedule_sampler.sample(micro.shape[0], device)

                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )
            # -----------------------------------------------------------------

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            # Update sampler only for diffusion (FM usually uses simple uniform sampling)
            if isinstance(self.schedule_sampler, LossAwareSampler) and not self.flow_matching:
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.ls = loss.item()
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step + self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"), "wb"
        ) as f:
            th.save(self.opt.state_dict(), f)

    # def save(self):
    #     def save_checkpoint(rate, params):
    #         state_dict = self.mp_trainer.master_params_to_state_dict(params)
    #         if dist.get_rank() == 0:
    #             logger.log(f"saving model {rate}...")
    #             if not rate:
    #                 filename = f"model{(self.step+self.resume_step):06d}.pt"
    #             else:
    #                 filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
    #             with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
    #                 th.save(state_dict, f)

    #     save_checkpoint(0, self.mp_trainer.master_params)
    #     for rate, params in zip(self.ema_rate, self.ema_params):
    #         save_checkpoint(rate, params)

    #     if dist.get_rank() == 0:
    #         with bf.BlobFile(
    #             bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
    #             "wb",
    #         ) as f:
    #             th.save(self.opt.state_dict(), f)

    #     dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
