# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: speech dereverberation and blind RIR identification.

from glob import glob
from tqdm import tqdm
import os
import torch
from collections import defaultdict
import toml
from typing import Dict
from jsonargparse import ArgumentParser
import json
from trainer_inferencer.utils import initialize_module
from pathlib import Path


def save_config_to_file(args, file_path):
    with open(file_path, "w") as json_file:
        json.dump(args.__dict__, json_file, indent=4)


def average_checkpoints(checkpoints):
    param_sums = defaultdict(lambda: 0)
    num_checkpoints = len(checkpoints)
    for ckpt in checkpoints:
        if "use_ema" in ckpt and ckpt["use_ema"]:
            state_dict = ckpt["model_ema"]
        else:
            state_dict = ckpt["model"]

        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            param_sums[new_key] += value.float()

    averaged_state_dict = {}
    for key, sum_value in param_sums.items():
        averaged_state_dict[key] = sum_value / num_checkpoints

    return averaged_state_dict


@torch.no_grad()
def inference(
    input_path: str,
    output_path: str,
    model: Dict,
    EM_algo: Dict,
    acoustic: Dict,
    ckpt: str,
    device: str,
    *args,
    **kwargs
):

    fpath_input = sorted(
        glob("{}/**/*{}".format(input_path, ".flac"), recursive=True)
    ) + sorted(glob("{}/**/*{}".format(input_path, ".wav"), recursive=True))
    basename_input = [os.path.basename(i) for i in fpath_input]
    N_seq = len(basename_input)

    TF = initialize_module(acoustic["path"], acoustic["args"])
    sr = TF.sr

    mymodel = initialize_module(model["path"], model["args"])
    mymodel.to(device)

    ckpt = torch.load(ckpt, map_location=device)

    new_state_dict = {}
    for k, v in ckpt["model"].items():
        if any(x in k for x in ["ops", "params"]):
            continue
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    mymodel.load_state_dict(new_state_dict, strict=True)
    mymodel.eval()

    pim = initialize_module(EM_algo["path"], EM_algo["args"])

    TF = initialize_module(acoustic["path"], acoustic["args"])
    for fpath_input_n in tqdm(fpath_input):
        basename_input_n = os.path.basename(fpath_input_n)
        fpath_out_rir = os.path.join(output_path, "rir", basename_input_n)

        input_wav = TF.load_wav(fpath_input_n, sr).to(device)

        rir = pim.process(input_wav, mymodel, TF, device)

        TF.save_wav(rir / rir.abs().max(), fpath_out_rir, sr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Configuration .toml file"
    )
    parser.add_argument("--ckpt", required=True, type=str, help="checkpoint path")
    parser.add_argument(
        "-i", "--input_path", required=True, type=str, help="input path"
    )
    parser.add_argument(
        "-o", "--output_path", required=True, type=str, help="output path"
    )
    parser.add_argument("-d", "--device", required=False, type=str, default="cuda:0")

    args = parser.parse_args()

    config_path = Path(args.config).expanduser().absolute()
    os.makedirs(args.output_path, exist_ok=True)
    config = toml.load(config_path.as_posix())
    with open(os.path.join(args.output_path, "config.toml"), "w") as f:
        toml.dump(config, f)

    save_config_to_file(args, os.path.join(args.output_path, "config.json"))

    inference(**args, **config)

    """
    usage:
    python enhance_rir_avg.py -c [config filepath] --ckpt [checkpoint filepath list] -i [reverberant speech dirpath] -o [output dirpath] -d [device ID]
    """
