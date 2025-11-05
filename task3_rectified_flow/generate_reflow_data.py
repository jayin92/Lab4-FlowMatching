"""
Generate Reflow Dataset for Rectified Flow Training (Task 3)

This script generates synthetic training pairs (x_0, z_1) by simulating the learned flow
from a pretrained Flow Matching model. These pairs are used to train a rectified flow model
that learns straighter trajectories.

Reference: Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.append('.')
from image_common.dataset import tensor_to_pil_image
from image_common.fm import FlowMatching


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save configuration
    config = vars(args)
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    device = f"cuda:{args.gpu}"

    # Load pretrained Flow Matching model
    print(f"Loading checkpoint from {args.ckpt_path}")
    fm = FlowMatching(None, None)
    fm.load(args.ckpt_path)
    fm.eval()
    fm = fm.to(device)

    if args.use_cfg:
        assert fm.network.use_cfg, "The model was not trained with CFG support."
        num_classes = fm.network.class_embedding.num_embeddings - 1
        print(f"Using CFG with {num_classes} classes (including null class)")

    total_num_samples = args.num_samples
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    print(f"Generating {total_num_samples} reflow pairs with {args.num_inference_steps} ODE steps...")

    for i in tqdm(range(num_batches), desc="Generating reflow dataset"):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        B = eidx - sidx

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Implement reflow data generation:
        # 1. Sample x_0 from prior distribution (e.g., standard Gaussian)
        # 2. Sample class labels for conditional generation (if using CFG)
        # 3. Simulate the learned ODE from t=0 to t=1 to obtain z_1 = Φ_1(x_0)
        # 4. Save the pairs (x_0, z_1, labels) to disk
        #
        # Hint: Use fm.sample() to simulate the flow, but you need to store both
        # the initial noise x_0 and final generated sample z_1.
        # Hint: Set return_traj=True to get the full trajectory if needed.

        # Sample initial noise
        shape = (B, 3, fm.image_resolution, fm.image_resolution)
        x_0 = torch.randn(shape).to(device)

        # Sample class labels if using CFG
        if args.use_cfg:
            # Sample labels from 1 to num_classes-1 (skip null class 0)
            labels = torch.randint(1, num_classes, (B,)).to(device)
        else:
            labels = None

        # Generate z_1 by simulating the learned flow
        # TODO: Complete this section
        # Use fm.sample() or manually implement the ODE integration
        # to generate z_1 from x_0

        z_1 = x_0  # Replace this with actual generation

        ######################

        # Save the pairs to disk
        for j in range(B):
            sample_idx = sidx + j

            # Save as .pt files for efficient loading
            torch.save(x_0[j].cpu(), save_dir / f"{sample_idx:06d}_x0.pt")
            torch.save(z_1[j].cpu(), save_dir / f"{sample_idx:06d}_z1.pt")

            if args.use_cfg:
                torch.save(labels[j].cpu(), save_dir / f"{sample_idx:06d}_label.pt")

            # Optionally save as images for visualization
            if args.save_images and sample_idx < 100:  # Save first 100 as images
                img_z1 = tensor_to_pil_image(z_1[j].cpu(), single_image=True)
                img_z1.save(save_dir / f"{sample_idx:06d}_z1.png")

    print(f"Reflow dataset saved to {save_dir}")
    print(f"Total samples: {total_num_samples}")

    # Save metadata
    metadata = {
        "num_samples": total_num_samples,
        "num_inference_steps": args.num_inference_steps,
        "use_cfg": args.use_cfg,
        "guidance_scale": args.cfg_scale if args.use_cfg else None,
        "image_resolution": fm.image_resolution,
        "num_classes": num_classes if args.use_cfg else None,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reflow dataset for Rectified Flow training")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to pretrained FM checkpoint")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of reflow pairs to generate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for generation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--save_dir", type=str, default="data/afhq_reflow", help="Directory to save reflow dataset")
    parser.add_argument("--use_cfg", action="store_true", help="Use classifier-free guidance")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of ODE steps for generation")
    parser.add_argument("--save_images", action="store_true", help="Save first 100 samples as images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)
