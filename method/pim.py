# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: VEM algorithm.

import torch
from torch import vmap
from torch.nn.functional import pad
import numpy as np
import torchaudio
import soundfile as sf


class PIM:
    def __init__(self, sr=16000, *args, **kwargs):
        """
        Initialization
        """
        self.sr = sr

        pi = np.pi
        duration = 8.192
        f1 = 62.5
        f2 = self.sr / 2
        w1 = 2 * pi * f1 / self.sr
        w2 = 2 * pi * f2 / self.sr
        num_sample = int(duration * self.sr)
        sinesweep = np.zeros(num_sample)

        taxis = np.arange(0, num_sample, 1) / (num_sample - 1)

        lw = np.log(w2 / w1)
        sinesweep = np.sin(w1 * (num_sample - 1) / lw * (np.exp(taxis * lw) - 1))

        envelope = (w2 / w1) ** (-taxis)

        invfilter = np.flipud(sinesweep) * envelope
        scaling = pi * num_sample * (w1 / w2 - 1) / (2 * (w2 - w1) * np.log(w1 / w2))
        invfilter = invfilter / scaling

        sinesweep = self.apply_ramp(
            sinesweep, left_ramp_sample=256, right_ramp_sample=128
        )

        self.sinesweep = pad(torch.from_numpy(sinesweep).float(), (512, 512))

        self.invfilter = pad(torch.from_numpy(invfilter).float(), (512, 512))

    def init_seg(self, TF, model, Obs_wav, *args, **kwargs):
        """
        对每段音频的初始化
        input:
            Obs: [F,T] complex
        return:
            Obs: [F,T] complex 观测频谱
            Sig_var: [F,T] real 干净语音先验方差
            Noi_var: [F,T] real 噪声先验方差
            CTF: [F,L] complex CTF滤波器
            Err_var: [F] real 误差方差
        """

        self.device = Obs_wav.device

        Obs_wav = Obs_wav.squeeze()
        Obs_wav_normed = TF.norm_amplitude(Obs_wav)
        Obs = TF.stft(Obs_wav_normed, "complex")  # [F,T] complex
        F, T = Obs.shape
        self.dtype = Obs.dtype

        _, CTF_ft, _ = model(TF.preprocess(Obs.unsqueeze(0).unsqueeze(1)))
        CTF = TF.postprocess(CTF_ft).squeeze()
        CTF=CTF.flip(-1)
        L = CTF.shape[-1]

        return CTF

    @torch.no_grad()
    def process(self, Obs_wav, model, TF, device, *args, **kwargs):

        CTF = self.init_seg(TF, model, Obs_wav)
        L = CTF.shape[-1]

        sinesweep = self.sinesweep.to(device)
        invfilter = self.invfilter.to(device)
        sinesweep_spec = TF.stft(sinesweep, "complex")

        CTF_ret = CTF.unsqueeze(2)

        sinesweep_spec = pad(sinesweep_spec, (L - 1, L - 1))  # [F,T+L-1]
        sinesweep_spec = sinesweep_spec.unfold(1, L, 1)  # [F,T,L] complex

        ir_spec = torch.matmul(sinesweep_spec, CTF_ret).squeeze()
        ir = TF.istft(ir_spec, "complex")

        rir = torchaudio.functional.convolve(invfilter, ir, mode="full").to("cpu")

        rir = rir[torch.argmax(rir.abs()) - int(self.sr * 0.0025) :]
        if rir.abs().max() > 1:
            rir /= rir.abs().max()

        return rir[:self.sr*2]

    def apply_ramp(self, signal, left_ramp_sample=512, right_ramp_sample=512):

        n_samples = len(signal)
        left_ramp_length = left_ramp_sample
        right_ramp_length = right_ramp_sample

        left_ramp = np.hanning(left_ramp_length * 2)[:left_ramp_length]
        right_ramp = np.hanning(right_ramp_length * 2)[:right_ramp_length]

        output = signal.copy()
        output[:left_ramp_length] *= left_ramp
        output[-right_ramp_length:] *= right_ramp[::-1]

        return output
