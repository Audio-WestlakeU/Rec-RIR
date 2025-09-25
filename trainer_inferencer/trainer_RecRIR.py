import torch
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from .base_trainer_RecRIR import BaseTrainer
from .utils import plot_spectrogram
import torchaudio
from model.lossF import RIMag_loss, MSE_loss_complex
from torch.amp import GradScaler, autocast

lossF_RIMag = RIMag_loss()
lossF_MSE = MSE_loss_complex()


class Trainer(BaseTrainer):
    def __init__(
        self,
        dist,
        rank,
        config,
        resume: bool,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        valid_dataloader,
        start_ckpt,
        *args,
        **kwargs,
    ):
        super().__init__(
            dist,
            rank,
            config,
            resume,
            model,
            optimizer,
            scheduler,
            start_ckpt,
        )

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.loss_w_rvb = config["trainer"]["train"]["loss_w_rvb"]
        self.loss_w_cln = config["trainer"]["train"]["loss_w_cln"]
        self.loss_w_rec = config["trainer"]["train"]["loss_w_rec"]

        self.scaler = GradScaler()
        if config["trainer"]["train"]["loss_type"] == "RIMag":
            self.lossF = lossF_RIMag
        elif config["trainer"]["train"]["loss_type"] == "MSE":
            self.lossF = lossF_MSE

    def _train_epoch(self, epoch):

        loss_total = 0.0
        self.optimizer.zero_grad()
        for index, (noisy_wav, rev_wav, dp_wav, fpath) in (
            enumerate(tqdm(self.train_dataloader, desc="Training"))
            if self.rank == 0
            else enumerate(self.train_dataloader)
        ):
            self.optimizer.zero_grad()
            with autocast(dtype=torch.float16, device_type="cuda"):
                noisy_wav = noisy_wav.to(self.rank)
                rev_wav = rev_wav.to(self.rank)
                dp_wav = dp_wav.to(self.rank)

                input_complex = self.transformfunc.stft(
                    noisy_wav, output_type="complex"
                ).to(dtype=torch.complex64)
                target_complex = self.transformfunc.stft(
                    dp_wav, output_type="complex"
                ).to(dtype=torch.complex64)
                reverb_complex = self.transformfunc.stft(
                    rev_wav, output_type="complex"
                ).to(dtype=torch.complex64)

                input_ft = self.transformfunc.preprocess(input_complex)

                est_spch_ft, est_ctf_ft, est_reverb_ft = self.compiled_model(input_ft)

                est_spch = self.transformfunc.postprocess(est_spch_ft).to(
                    dtype=torch.complex64
                )
                est_ctf = self.transformfunc.postprocess(est_ctf_ft).to(
                    dtype=torch.complex64
                )
                est_reverb = self.transformfunc.postprocess(est_reverb_ft).to(
                    dtype=torch.complex64
                )
                L = est_ctf.shape[-1]
                recon = torchaudio.functional.convolve(
                    target_complex, est_ctf, mode="full"
                )[..., : target_complex.shape[-1]]

                ######## calculate losses ########
                loss_cln = self.lossF(est_spch, target_complex)
                loss_rvb = self.lossF(est_reverb, reverb_complex)
                loss_rec = self.lossF(recon, reverb_complex)

                loss_gd = (
                    self.loss_w_cln * loss_cln
                    + self.loss_w_rec * loss_rec
                    + self.loss_w_rvb * loss_rvb
                )

            self.scaler.scale(loss_gd).backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_norm_value
            )

            self.scaler.step(self.optimizer)

            self.scaler.update()
            self.steps += 1

            if self.scheduler:
                self.scheduler.step(self.steps)

            loss_total = loss_total + loss_gd.item()

            if (index) % 100 == 0:

                if self.rank == 0:
                    self.writer.add_scalar(
                        f"Train_step/Lr",
                        self.optimizer.param_groups[0]["lr"],
                        self.steps,
                    )
                    self.writer.add_scalar(
                        f"Train_step/Loss",
                        loss_gd.item(),
                        self.steps,
                    )
                    self.writer.add_scalar(
                        f"Train_step/loss_cln",
                        loss_cln.item(),
                        self.steps,
                    )
                    self.writer.add_scalar(
                        f"Train_step/loss_rvb",
                        loss_rvb.item(),
                        self.steps,
                    )
                    self.writer.add_scalar(
                        f"Train_step/loss_rec",
                        loss_rec.item(),
                        self.steps,
                    )

        if self.rank == 0:
            self.writer.add_scalar(
                f"Train_epoch/Loss",
                loss_total / len(self.train_dataloader),
                epoch,
            )

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0
        loss_total_cln = 0.0
        loss_total_rvb = 0.0
        loss_total_rec = 0.0

        for index, (noisy_wav, rev_wav, dp_wav, fpath) in (
            enumerate(tqdm(self.valid_dataloader, desc="Validating"))
            if self.rank == 0
            else enumerate(self.valid_dataloader)
        ):
            noisy_wav = noisy_wav.to(self.rank)
            rev_wav = rev_wav.to(self.rank)
            dp_wav = dp_wav.to(self.rank)

            input_complex = self.transformfunc.stft(noisy_wav, output_type="complex")
            target_complex = self.transformfunc.stft(dp_wav, output_type="complex")
            reverb_complex = self.transformfunc.stft(rev_wav, output_type="complex")

            input_ft = self.transformfunc.preprocess(input_complex)
            est_spch_ft, est_ctf_ft, est_reverb_ft = self.model(input_ft)

            est_spch = self.transformfunc.postprocess(est_spch_ft)
            est_ctf = self.transformfunc.postprocess(est_ctf_ft)
            est_reverb = self.transformfunc.postprocess(est_reverb_ft)

            L = est_ctf.shape[-1]
            recon = torchaudio.functional.convolve(
                target_complex, est_ctf, mode="full"
            )[..., : target_complex.shape[-1]]

            ######## calculate losses ########
            loss_cln = self.lossF(est_spch, target_complex)
            loss_rvb = self.lossF(est_reverb, reverb_complex)
            loss_rec = self.lossF(recon, reverb_complex)

            loss = (
                self.loss_w_cln * loss_cln
                + self.loss_w_rec * loss_rec
                + self.loss_w_rvb * loss_rvb
            )

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_cln, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_rec, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_rvb, op=dist.ReduceOp.SUM)

            loss_total += loss
            loss_total_cln += loss_cln
            loss_total_rvb += loss_rvb
            loss_total_rec += loss_rec
            if self.rank == 0:
                self.writer.add_scalar(
                    f"Valid_epoch/Loss",
                    loss_total / len(self.valid_dataloader) / dist.get_world_size(),
                    epoch,
                )
                self.writer.add_scalar(
                    f"Valid_epoch/loss_cln",
                    loss_total_cln / len(self.valid_dataloader) / dist.get_world_size(),
                    epoch,
                )
                self.writer.add_scalar(
                    f"Valid_epoch/loss_rvb",
                    loss_total_rvb / len(self.valid_dataloader) / dist.get_world_size(),
                    epoch,
                )
                self.writer.add_scalar(
                    f"Valid_epoch/loss_rec",
                    loss_total_rec / len(self.valid_dataloader) / dist.get_world_size(),
                    epoch,
                )

                if index == 0:
                    self.writer.add_figure(
                        f"Valid_epoch/target_cln",
                        plot_spectrogram(target_complex[0].abs().squeeze()),
                    )
                    self.writer.add_figure(
                        f"Valid_epoch/input",
                        plot_spectrogram(input_complex[0].abs().squeeze()),
                    )
                    self.writer.add_figure(
                        f"Valid_epoch/target_reverb",
                        plot_spectrogram(reverb_complex[0].abs().squeeze()),
                    )
                    self.writer.add_figure(
                        f"Valid_epoch/est_cln",
                        plot_spectrogram(est_spch[0].abs().squeeze()),
                    )
                    self.writer.add_figure(
                        f"Valid_epoch/est_reverb",
                        plot_spectrogram(recon[0].abs().squeeze()),
                    )
                    self.writer.add_figure(
                        f"Valid_epoch/est_ctf",
                        plot_spectrogram(est_ctf[0].abs().squeeze()),
                    )

        return loss_total / len(self.valid_dataloader) / dist.get_world_size()
