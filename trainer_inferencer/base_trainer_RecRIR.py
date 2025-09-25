# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: trainer base class.

import os
from os import path
import toml
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from .utils import initialize_module


class BaseTrainer:

    def __init__(
        self,
        dist,
        rank,
        config,
        resume: bool,
        model,
        optimizer,
        scheduler,
        start_ckpt: None,
    ):

        self.dist = dist
        self.rank = rank

        self.model = DDP(
            model.cuda(rank), device_ids=[rank], find_unused_parameters=False
        )
        self.compiled_model = torch.compile(self.model)

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dataloader = None

        num_gpus = int(os.environ["WORLD_SIZE"])
        config["dataloader"]["args"]["batchsize"][0] *= num_gpus
        config["dataloader"]["args"]["batchsize"][1] *= num_gpus

        # meta config
        self.meta_config = config["meta"]
        torch.backends.cudnn.enabled = self.meta_config["cudnn_enable"]
        self.save_dir = self.meta_config["save_dir"]
        self.ckpt_dir = path.join(self.save_dir, "ckpt")
        self.log_dir = path.join(self.save_dir, "log")

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # acoustic config
        self.acoustic_config = config["acoustic"]
        self.transformfunc = initialize_module(
            self.acoustic_config["path"], self.acoustic_config["args"]
        )
        self.sr = self.transformfunc.sr

        # training config
        self.train_config = config["trainer"]["train"]
        self.epochs = self.train_config["epochs"]
        self.save_ckpt_interval = self.train_config["save_ckpt_interval"]
        self.clip_grad_norm_value = self.train_config["clip_grad_norm_value"]

        # validation config
        self.valid_config = config["trainer"]["validation"]
        self.valid_interval = self.valid_config["interval"]
        self.save_max_metric = self.valid_config["save_max_metric"]

        # logger
        self.logger = logging.getLogger("mylogger")
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(path.join(self.log_dir, "log.txt"), mode="a")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # initialization
        self.start_epoch = 1
        self.best_metric = -torch.inf if self.save_max_metric else torch.inf
        self.steps = 0
        if resume:
            self._resume_ckpt()

        if start_ckpt:
            ckpt = torch.load(start_ckpt, map_location="cpu")

            self.dist.barrier()
            ckpt["model"] = {
                k: v
                for k, v in ckpt["model"].items()
                if not any(x in k for x in ["ops", "params"])
            }
            self.model.load_state_dict(ckpt["model"], strict=False)

        if self.rank == 0:
            self.writer = SummaryWriter(self.log_dir)
            self.writer.add_text(
                tag="config",
                text_string=f"<pre>  \n{toml.dumps(config)}  \n</pre>",
                global_step=1,
            )
            with open(
                path.join(self.save_dir, f"{time.strftime('%Y-%m-%d-%H-%M-%S')}.toml"),
                "w",
            ) as handle:
                toml.dump(config, handle)

    def _save_best_latest_ckpt(
        self, epoch: int, is_best: bool = False, period: bool = False
    ):
        torch.cuda.synchronize()
        state_dict = {
            "epoch": epoch,
            "best_metric": self.best_metric,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "steps": self.steps,
        }

        state_dict["model"] = self.model.state_dict()

        torch.save(state_dict, path.join(self.ckpt_dir, f"latest.tar"))
        if is_best:
            self.logger.info(f"New best model saved")
            torch.save(state_dict, path.join(self.ckpt_dir, f"best.tar"))
        if period:
            torch.save(state_dict, path.join(self.ckpt_dir, f"epoch{epoch}.tar"))

    def _is_best(self, metric):
        """
        check if the checkpoint is the best
        """
        if self.save_max_metric and metric > self.best_metric:
            self.best_metric = metric
            return True
        elif not self.save_max_metric and metric < self.best_metric:
            self.best_metric = metric
            return True
        else:
            return False

    def _resume_ckpt(self):
        """resume training"""
        latest_model_path = path.join(self.ckpt_dir, "latest.tar")
        assert path.exists(latest_model_path), f"{latest_model_path} does not exist"

        ckpt = torch.load(latest_model_path, map_location="cpu")

        self.dist.barrier()

        self.start_epoch = int(ckpt["epoch"] + 1)
        self.best_metric = ckpt["best_metric"]
        self.steps = ckpt["steps"]

        ckpt["model"] = {
            k: v
            for k, v in ckpt["model"].items()
            if not any(x in k for x in ["ops", "params"])
        }
        self.model.load_state_dict(ckpt["model"], strict=False)

        if self.rank == 0:
            self.logger.info(
                f"Model checkpoint is loaded. Training will begin at epoch {self.start_epoch}."
            )

    def _set_train_mode(self):
        self.compiled_model.train()
        self.model.train()

    def _set_valid_mode(self):
        torch.cuda.synchronize()
        self.compiled_model.eval()
        self.model.eval()

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _validation_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        metric = torch.inf

        for epoch in range(self.start_epoch, self.epochs + 1):

            self.train_dataloader.sampler.set_epoch(epoch)
            if self.rank == 0:
                self.logger.info(f"{'=' * 5} epoch {epoch} {'=' * 5}")
            self._set_train_mode()
            self._train_epoch(epoch)

            if self.rank == 0:
                self.writer.add_scalar(
                    f"Lr",
                    self.optimizer.param_groups[0]["lr"],
                    epoch,
                )

            if epoch % self.valid_interval == 0:
                self._set_valid_mode()
                metric = self._validation_epoch(epoch)
                if self.rank == 0:
                    if epoch % self.save_ckpt_interval == 0:
                        self._save_best_latest_ckpt(epoch, is_best=False, period=True)
                    self._save_best_latest_ckpt(epoch, is_best=self._is_best(metric))
