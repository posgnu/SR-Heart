from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import argparse
import torchvision.transforms as T
import torch



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="1")
    parser.add_argument("--sample-num", type=int, default=8)


    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    opt = parse_opt()

    model = Unet(
        channels=1,
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
        'data',
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
    )

    trainer.load(opt.weights)

    sampled_images = diffusion.sample(batch_size = opt.sample_num)

    transform = T.ToPILImage()
    img = transform(torch.cat([image for image in sampled_images], dim=2))
    img.save("sample.png")


