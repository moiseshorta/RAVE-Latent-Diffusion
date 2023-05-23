#### RAVE-Latent Diffusion
#### https://github.com/moiseshorta/RAVE-Latent-Diffusion
####
#### Author: Moises Horta Valenzuela / @hexorcismos
#### Year: 2023
#### Description:
#### This script generates audio using a denoising diffusion model and a pretrained RAVE model.
#### The script takes as input a pretrained diffusion model, a pretrained RAVE model, the number of diffusion steps,
#### a random seed, an output path, and a name for the generated audio.
#### It generates a specified number of audio files with the provided settings.
#### The generation process involves creating latent codes with the diffusion model,
#### decoding them with the RAVE model, and saving the resulting audio to the specified output path.

import argparse
import os
import torch
import numpy as np
import random
import soundfile as sf
from tqdm import tqdm
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parse the input arguments for the script.
def parse_args():
    parser = argparse.ArgumentParser(description="Generate RAVE latents using diffusion model.")
    parser.add_argument("--model_path", type=str, required=True, default=None, help="Path to the pretrained diffusion model checkpoint.")
    parser.add_argument("--rave_model", type=str, required=True, default=None, help="Path to the pretrained RAVE model (.ts).")
    parser.add_argument("--rave_dims", type=int, choices=[4, 8, 16, 32, 64, 128], default=16, help="Number of hidden dimensions in RAVE model.")
    parser.add_argument("--sample_rate", type=int, default=48000, choices=[44100, 48000], help="Sample rate for generated audio. Should match samplerate of RAVE model.")
    parser.add_argument("--diffusion_steps", type=int, default=100, help="Number of steps for denoising diffusion.")
    parser.add_argument("--seed", type=int, default=random.randint(0,2**31-1), help="Random seed for generation.")
    parser.add_argument('--latent_length', type=int, default=4096, choices=[2048, 4096, 8192, 16384], help='Length of generated RAVE latents.')
    parser.add_argument("--length_mult", type=int, default=1, help="Multiply the duration of output by default model window.")
    parser.add_argument("--output_path", type=str, default="./", help="Path to the output audio file.")
    parser.add_argument("--num", type=int, default=1, help="Number of audio to generate.")
    parser.add_argument("--name", type=str, default="out", help="Name of audio to generate.")
    parser.add_argument("--lerp", type=bool, default=False, help="Interpolate between two seeds.")
    parser.add_argument("--lerp_factor", type=float, default=1.0, help="Interpolating factor between two seeds.")
    parser.add_argument("--seed_a", type=int, default=random.randint(0,2**31-1), help="Starting seed for interpolation.")
    parser.add_argument("--seed_b", type=int, default=random.randint(0,2**31-1), help="Ending seed for interpolation.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature of the random noise before diffusion.")
    return parser.parse_args()

def slerp(val, low, high):
    omega = torch.acos((low/torch.norm(low, dim=2, keepdim=True) * high/torch.norm(high, dim=2, keepdim=True)).sum(dim=2, keepdim=True).clamp(-1, 1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so) * low + (torch.sin(val*omega)/so) * high
    return res

# Generate the audio using the provided models and settings.
def generate_audio(model, rave, args, seed):
    with torch.no_grad():
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        z_length = args.latent_length * args.length_mult

        noise = torch.randn(1, args.rave_dims, z_length).to(device)
        noise = noise * args.temperature

        rave_model_name = os.path.basename(args.rave_model).split(".")[0]
        diffusion_model_name = os.path.basename(args.model_path)

        print(f"Generating {z_length} latent codes with Diffusion model:", diffusion_model_name)
        print("Decoding using RAVE Model:", rave_model_name)
        print("Seed:", seed)

        model.eval()

        num_steps = 1
        with tqdm(total=num_steps) as pbar:
            for i in range(num_steps):
                ### GENERATING WITH .PT FILE
                diff = model.sample(noise, num_steps=args.diffusion_steps)
                # diff = model(noise)
                # noise = diff
                pbar.update(1)

        diff_mean = diff.mean()
        diff_std = diff.std()
        diff = (diff - diff_mean) / diff_std

        rave = rave.cpu()
        diff = diff.cpu()
        y = rave.decode(diff)

        y = y.reshape(-1).detach().numpy()

        y_left  = y[:len(y)//2]
        y_right = y[len(y)//2:]

        y_stereo = np.stack((y_left, y_right), axis=-1)
        sf.write(f'{args.output_path}/rave-latent_diffusion_seed{seed}_{args.name}_{rave_model_name}.wav', y_stereo, 48000)

# Generate audio by slerping between two diffusion generated RAVE latents.
def interpolate_seeds(model, rave, args, seed):
    with torch.no_grad():
        torch.manual_seed(seed)

        z_length = args.latent_length * args.length_mult

        torch.manual_seed(args.seed_a)
        noise1 = torch.randn(1, args.rave_dims, z_length).to(device) * args.temperature
        torch.manual_seed(args.seed_b)
        noise2 = torch.randn(1, args.rave_dims, z_length).to(device) * args.temperature

        rave_model_name = os.path.basename(args.rave_model).split(".")[0]
        diffusion_model_name = os.path.basename(args.model_path)

        print(f"Generating {z_length} latent codes with Diffusion model:", os.path.basename(args.model_path))
        print("Decoding using RAVE Model:", os.path.basename(args.rave_model))
        print("Interpolating with factor", args.lerp_factor)
        print("Seed A:", args.seed_a)
        print("Seed B:", args.seed_b)

        model.eval()

        num_steps = 1
        with tqdm(total=num_steps) as pbar:
            for i in range(num_steps):
                diff1 = model.sample(noise1, num_steps=args.diffusion_steps)
                diff2 = model.sample(noise2, num_steps=args.diffusion_steps)
                diff = slerp(torch.linspace(0., args.lerp_factor, z_length), diff1, diff2)
                pbar.update(1)

        diff_mean = diff.mean()
        diff_std = diff.std()
        diff = (diff - diff_mean) / diff_std

        rave = rave.cpu()
        diff = diff.cpu()
        y = rave.decode(diff)

        y = y.reshape(-1).detach().numpy()

        y_left  = y[:len(y)//2]
        y_right = y[len(y)//2:]

        y_stereo = np.stack((y_left, y_right), axis=-1)
        sf.write(f'{args.output_path}/rave-latent_diffusion_seed{seed}_{args.name}_{rave_model_name}_slerp.wav', y_stereo, 48000)

# Main function sets up the models and generates the audio.
def main():
    args = parse_args()

    rave = torch.jit.load(args.rave_model).to(device)

    ### GENERATING WITH .PT FILE DIFFUSION
    model = DiffusionModel(
        net_t=UNetV0,
        in_channels=16,
        channels=[256, 256, 256, 256, 512, 512, 512, 768, 768],
        factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],
        items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
        attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],
        attention_heads=12,
        attention_features=64,
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if not args.lerp:
        for i in range(args.num):
            seed = args.seed + i
            generate_audio(model, rave, args, seed)
    else:
        for i in range(args.num):
            seed = args.seed + i
            interpolate_seeds(model, rave, args, seed)

if __name__ == "__main__":
    main()