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
import torch
import librosa as li
import os
import numpy as np
from torch.utils.data import Dataset
from pydub import AudioSegment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rave_model', type=str, default='/path/to/rave_model', help='Path to the exported (.ts) RAVE model.')
    parser.add_argument('--audio_folder', type=str, default='/path/to/audio_folder', help='Path to the folder containing audio files.')
    parser.add_argument('--sample_rate', type=int, default=48000, choices=[44100, 48000], help='Sample rate for the audio files.')
    parser.add_argument('--latent_length', type=int, default=4096, choices=[2048, 4096, 8192, 16384], help='Length of saved RAVE latents.')
    parser.add_argument('--latent_folder', type=str, default='/path/to/latent_folder', help='Path to the folder where RAVE latent files will be saved.')
    return parser.parse_args()

def encode_and_save_latent(rave, audio_data, audio_file, latent_folder, latent_length):
    with torch.no_grad():

        x = torch.from_numpy(audio_data).reshape(1, 1, -1).float()

        print("Audio",audio_file)

        if device.type == 'cuda':
            x = x.to(device)
            rave = rave.to(device)

        z = rave.encode(x)
        print("Encoded into latent",z.shape)

        z_mean = z.mean()
        z_std = z.std()
        z = (z - z_mean) / z_std

        if device.type == 'cuda':
            z = z.cpu()

        z = torch.nn.functional.pad(z, (0, latent_length - z.shape[2]))

        z = z.detach().numpy()

        print("Saving latent of shape", z.shape)

        np.save(os.path.join(latent_folder, audio_file[:-4] + '.npy'), z)

def main():
    args = parse_args()

    rave = torch.jit.load(args.rave_model).to(device)

    audio_files = [f for f in os.listdir(args.audio_folder) if f.endswith(".wav")]

    if args.sample_rate == 44100:
        if args.latent_length == 2048:
            crop_duration = 96 * 1000

        elif args.latent_length == 4096:
            crop_duration = 192 * 1000

        elif args.latent_length == 8192:
            crop_duration = 384 * 1000

        elif args.latent_length == 16384:
            crop_duration = 768 * 1000

    if args.sample_rate == 48000:
        if args.latent_length == 2048:
            crop_duration = 88 * 1000

        elif args.latent_length == 4096:
            crop_duration = 176 * 1000

        elif args.latent_length == 8192:
            crop_duration = 352 * 1000

        elif args.latent_length == 16384:
            crop_duration = 704 * 1000

    for audio_file in audio_files:
        full_path = os.path.join(args.audio_folder, audio_file)
        base_name = os.path.splitext(audio_file)[0]
        audio = AudioSegment.from_wav(full_path)

        # Set the sample rate
        audio = audio.set_frame_rate(args.sample_rate)
        audio = audio.set_channels(1)

        file_duration = len(audio)  # Duration in milliseconds
        num_segments = file_duration // crop_duration

        for i in range(num_segments):
            start_time = i * crop_duration
            end_time = start_time + crop_duration
            cropped_audio = audio[int(start_time):int(end_time)]

            output_file = f"{base_name}_part{i:03d}.wav"

            # Convert pydub.AudioSegment to numpy array
            cropped_data = np.array(cropped_audio.get_array_of_samples())

            encode_and_save_latent(rave, cropped_data, output_file, args.latent_folder, args.latent_length)

    print('Done encoding RAVE latents')
    print('Path to latents:', args.latent_folder)

if __name__ == "__main__":
    main()
