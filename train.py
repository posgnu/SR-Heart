from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


if __name__ == "__main__":

    model = Unet(
        channels=3,
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l1'            # L1 or L2
    ).cuda()

    trainer = Trainer(
        diffusion,
        'data/LV',
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 700000,         # total training steps, [700000] 
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = False,                        # turn on mixed precision
    )

    trainer.train()

    sampled_images = diffusion.sample(batch_size = 4)
