# RAVE-Latent Diffusion
Generate new latent codes for RAVE using Denoising Diffusion Probabilistic models.

> Author: Moisés Horta Valenzuela / [`𝔥𝔢𝔵𝔬𝔯𝔠𝔦𝔰𝔪𝔬𝔰`](https://twitter.com/hexorcismos)
> 
> Year: 2023

RAVE-Latent Diffusion is a denoising diffusion model designed to generate new [`RAVE`](https://github.com/acids-ircam/RAVE) latent codes with a large context window, faster than realtime, while maintaining music structural coherency. RAVE-Latent Diffusion can currently be used for unconditional RAVE latent generation, with the smallest context window of aprox. 1:30 minutes of audio (```preprocess.py --latent_length=2048 ...```) and can scale up to a context window of 11:30 minutes (```preprocess.py --latent_length=16384 ...```).
RAVE-Latent Diffusion can generate audio faster than real-time on a consumer CPU, which allows for wide accesibility.

Audio examples can be found [`here`](https://soundcloud.com/h-e-x-o-r-c-i-s-m-o-s/rave-latentdiffusion_unconditionalgeneration_seed2805182108), [`here`](https://soundcloud.com/h-e-x-o-r-c-i-s-m-o-s/rave-latent-diffusion_unconditionalgeneration_seed3069861997) and [`here`](https://soundcloud.com/h-e-x-o-r-c-i-s-m-o-s/s-1).

## Prerequisites:

1) Pretrained RAVE model, exported as .ts file:
> Make sure that your pre-trained RAVE model has a latent dimension size of powers of two. The default RAVE latent dimension size is 16. For more information on how to modify this, see the original [`RAVE`](https://github.com/acids-ircam/RAVE) repo.

2) Audio dataset used to train RAVE model: 
> The recommended audio dataset length is 3 hours, but training on more data will prevent overfitting. You can also train on audio that has not been 'heard' by the pre-trained RAVE model, but ultimately the fidelity of the generated audio will depend on the reconstruction quality that your pre-trained RAVE model has on your audio dataset.

## Instructions

The process of training a new RAVE-Latent Diffusion model consists of three steps:
1) Pre-processing by converting your audio into RAVE latents of a fixed context window.
2) Training the RAVE-Latent Diffusion model.
3) Generate new latent codes with RAVE-Latent Diffusion and decode them into audio with a pretrained RAVE model.

## Install

Make a new Python virtual environment or conda environment. 
Execute the following commands:

```bash
git clone https://github.com/moiseshorta/RAVE-Latent-Diffusion.git
cd RAVE-Latent-Diffusion
pip install -r requirements.txt
```
Code has been tested on Python 3.9.

## Preprocessing

```bash
python preprocess.py --rave_model "/path/to/your/pretrained/rave/model.ts" --audio_folder "/path/to/your/audio/dataset" --latent_length 4096 --latent_folder "/path/to/save/encoded/rave/latents"
```
This is where you define your context window with the flag ```--latent_length```. The default value is ```4096``` latents context window, which is about 2:55 minutes of audio at 48KHz samplerate.
You can adjust the sample rate to match your pretrained RAVE model with the  ```--sample_rate``` flag. Currently it supports 44100 and 48000Hz sample rates.

## Training

Once you have preprocessed your audio dataset into RAVE latents, you can start training the latent diffusion model:

```bash
python train.py --name name_for_your_run --latent_folder "/path/to/saved/encoded/rave/latents" --save_out_path "/path/to/save/rave-latent-diffusion/checkpoints"
```

## Unconditional generation

Once you have trained the RAVE-Latent Diffusion model, you can now generate new latent codes with it. For example

```bash
python generate.py --model_path "/path/to/trained/rave-latent-diffusion/model.pt" --rave_model "/path/to/your/pretrained/rave/model.ts" --diffusion_steps 100 --seed 664 --output_path "/path/to/save/generated/audio" --latent_length 4096 --latent_mult 1
```
This command will generate a 4096 RAVE latents, defined by ```--latent_length=4096``` and multiply it by 1, defined by ```--latent_mult=1```. You can generate longer audio by modifying ```--latent_mult``` to 2 or another int. Be aware that your PC might run out of memory if the generated latent is too long. 
Keep in mind that RAVE-Latent Diffusion produces best results when the ```--latent_length``` is equal or higher than the context window used to train the; this value is defined by the ```--latent_length``` flag in the Preprocessing step.

## Spherical interpolation between generated RAVE latents

You can interpolate between two generated latent codes by setting the ```--lerp=True``` flag. This mode will generate two RAVE latent codes of ```--latent_length``` using two random seeds, slerp between them and decode them using a pretrained RAVE model.

```bash
python generate.py --model_path "/path/to/trained/rave-latent-diffusion/model.pt" --rave_model "/path/to/your/pretrained/rave/model.ts" --lerp True --diffusion_steps 100 --seed 664 --output_path "/path/to/save/generated/audio" --latent_length 4096 --latent_mult 1
```
For mode control, you can set the seed of the starting and ending RAVE latent codes with the ```--seed_a``` and ```--seed_b``` flags, respectively. 

## Text to audio generation

Coming soon...

## Audio in/out-painting

Coming soon...

## Credits

- This code builds on the work done at acids-ircam with [`RAVE (Caillon, 2022)`](https://arxiv.org/abs/2111.05011).
- The denoising diffusion model is based on the open-source [`audio-diffusion-pytorch`](https://github.com/archinetai/audio-diffusion-pytorch) library.
- Many thanks to Zach Evans from Harmon.ai for helping debug the code.

## Paper
Coming soon...
