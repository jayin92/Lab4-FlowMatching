"""
Training Script for Rectified Flow (Task 3 - Optional/Bonus)

Trains a rectified flow model on synthetic (x_0, z_1) pairs generated from a pretrained
Flow Matching model. The model learns straighter trajectories, enabling faster sampling.

Usage:
    python task3_rectified_flow/train_rectified.py \
        --reflow_data_path data/afhq_reflow \
        --use_cfg \
        --reflow_iteration 1
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from dotmap import DotMap
from fm import FlowMatching, FMScheduler
from network import UNet
from pytorch_lightning import seed_everything
from reflow_dataset import ReflowDataset, get_reflow_data_iterator
from dataset import tensor_to_pil_image
from tqdm import tqdm

matplotlib.use("Agg")


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"

    now = get_current_time()
    assert args.use_cfg, "In Assignment 7, we train with CFG setup only."

    # Create save directory with reflow iteration number
    if args.use_cfg:
        save_dir = Path(f"results/rectified_fm_{args.reflow_iteration}-{now}")
    else:
        save_dir = Path(f"results/rectified_fm_{args.reflow_iteration}_uncond-{now}")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"save_dir: {save_dir}")

    seed_everything(config.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    """######"""

    # Load reflow dataset
    print(f"Loading reflow dataset from {args.reflow_data_path}")
    reflow_dataset = ReflowDataset(
        args.reflow_data_path,
        use_cfg=args.use_cfg
    )

    train_dl = torch.utils.data.DataLoader(
        reflow_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )
    train_it = get_reflow_data_iterator(train_dl)

    # Get image resolution and num_classes from dataset metadata
    image_resolution = reflow_dataset.metadata.get("image_resolution", 64)
    num_classes = reflow_dataset.metadata.get("num_classes", None)

    print(f"Image resolution: {image_resolution}")
    if args.use_cfg:
        print(f"Number of classes: {num_classes}")

    # Set up the scheduler (same as base FM)
    fm_scheduler = FMScheduler(sigma_min=args.sigma_min)

    # Initialize network (same architecture as base FM)
    network = UNet(
        image_resolution=image_resolution,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
        num_classes=num_classes,
    )

    fm = FlowMatching(network, fm_scheduler)
    fm = fm.to(config.device)

    # Same optimizer and scheduler as base FM
    optimizer = torch.optim.Adam(fm.network.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
    )

    step = 0
    losses = []
    print(f"Starting training for {config.train_num_steps} steps...")
    print(f"This is {args.reflow_iteration}-rectified flow training")

    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:
            if step % config.log_interval == 0:
                fm.eval()
                # Plot loss curve
                plt.plot(losses)
                plt.savefig(f"{save_dir}/loss.png")
                plt.close()

                # Generate sample images
                shape = (4, 3, fm.image_resolution, fm.image_resolution)
                if args.use_cfg:
                    class_label = torch.tensor([1, 1, 2, 3]).to(config.device)
                    samples = fm.sample(
                        shape,
                        class_label=class_label,
                        guidance_scale=7.5,
                        num_inference_timesteps=20,
                        verbose=False
                    )
                else:
                    samples = fm.sample(shape, return_traj=False, verbose=False)

                pil_images = tensor_to_pil_image(samples)
                for i, img in enumerate(pil_images):
                    img.save(save_dir / f"step={step}-{i}.png")

                # Save checkpoint
                fm.save(f"{save_dir}/last.ckpt")
                fm.train()

            # Load batch from reflow dataset
            if args.use_cfg:
                x_0, z_1, label = next(train_it)
                x_0, z_1, label = x_0.to(config.device), z_1.to(config.device), label.to(config.device)
            else:
                x_0, z_1 = next(train_it)
                x_0, z_1 = x_0.to(config.device), z_1.to(config.device)
                label = None

            ######## TODO ########
            # DO NOT change the code outside this part.
            # Implement rectified flow training:
            # The key difference from base FM training is that we train on synthetic pairs (x_0, z_1)
            # instead of real data. The loss computation is identical to base FM.
            #
            # Hint: Use fm.get_loss() with x_0 as the prior and z_1 as the target.
            # The CFM loss is: E[||v_t(ψ_t(x_0|z_1); θ) - (z_1 - (1 - σ_min)x_0)||²]

            # Compute loss
            if args.use_cfg:
                loss = fm.get_loss(z_1, class_label=label, x0=x_0)
            else:
                loss = fm.get_loss(z_1, x0=x_0)

            ######################

            pbar.set_description(f"Loss: {loss.item():.4f}")

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            step += 1
            pbar.update(1)

    print(f"Training completed! Final checkpoint saved at {save_dir}/last.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Rectified Flow model")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--reflow_data_path", type=str, required=True, help="Path to reflow dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=100000,
        help="Number of training steps (same as base FM)",
    )
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--sigma_min", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--reflow_iteration", type=int, default=1, help="Reflow iteration number (1, 2, ...)")
    args = parser.parse_args()
    main(args)
