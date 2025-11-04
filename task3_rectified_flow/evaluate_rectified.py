"""
Evaluation Script for Rectified Flow (Task 3)

Evaluates rectified flow models with different numbers of sampling steps and
compares against base Flow Matching model.

Usage:
    python task3_rectified_flow/evaluate_rectified.py \
        --base_ckpt_path results/cfg_fm-XXX/last.ckpt \
        --rectified_ckpt_path results/rectified_fm_1-XXX/last.ckpt \
        --save_dir results/evaluation \
        --step_counts 5 10 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.append('.')
from image_common.dataset import tensor_to_pil_image
from image_common.fm import FlowMatching


def generate_samples(fm, num_samples, num_steps, batch_size, device, use_cfg=True, cfg_scale=7.5):
    """Generate samples with specified number of steps"""
    num_batches = int(np.ceil(num_samples / batch_size))
    all_samples = []

    start_time = time.time()

    for i in tqdm(range(num_batches), desc=f"Generating with {num_steps} steps"):
        sidx = i * batch_size
        eidx = min(sidx + batch_size, num_samples)
        B = eidx - sidx

        shape = (B, 3, fm.image_resolution, fm.image_resolution)

        if use_cfg:
            class_label = torch.randint(1, 4, (B,)).to(device)
            samples = fm.sample(
                shape,
                num_inference_timesteps=num_steps,
                class_label=class_label,
                guidance_scale=cfg_scale,
            )
        else:
            samples = fm.sample(shape, num_inference_timesteps=num_steps)

        all_samples.append(samples.cpu())

    elapsed_time = time.time() - start_time
    all_samples = torch.cat(all_samples, dim=0)[:num_samples]

    return all_samples, elapsed_time


def save_samples(samples, save_dir, prefix="sample"):
    """Save generated samples as images"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    pil_images = tensor_to_pil_image(samples)
    for i, img in enumerate(pil_images):
        img.save(save_dir / f"{prefix}_{i:04d}.png")


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = f"cuda:{args.gpu}"

    # Load models
    print(f"Loading base FM model from {args.base_ckpt_path}")
    fm_base = FlowMatching(None, None)
    fm_base.load(args.base_ckpt_path)
    fm_base.eval()
    fm_base = fm_base.to(device)

    print(f"Loading rectified flow model from {args.rectified_ckpt_path}")
    fm_rectified = FlowMatching(None, None)
    fm_rectified.load(args.rectified_ckpt_path)
    fm_rectified.eval()
    fm_rectified = fm_rectified.to(device)

    # Evaluation results
    results = {
        'Base FM': {},
        f'{args.reflow_iteration}-Rectified Flow': {}
    }

    print(f"\n{'='*80}")
    print(f"Evaluating with {args.num_samples} samples and CFG scale {args.cfg_scale}")
    print(f"{'='*80}\n")

    for num_steps in args.step_counts:
        print(f"\n{'='*60}")
        print(f"Evaluating with {num_steps} ODE steps")
        print(f"{'='*60}\n")

        # Generate samples with base FM
        print("Base FM:")
        samples_base, time_base = generate_samples(
            fm_base, args.num_samples, num_steps, args.batch_size, device,
            use_cfg=args.use_cfg, cfg_scale=args.cfg_scale
        )
        samples_per_sec_base = args.num_samples / time_base

        # Save samples
        base_save_dir = save_dir / f"base_fm_steps{num_steps}"
        save_samples(samples_base, base_save_dir)
        print(f"  Time: {time_base:.2f}s ({samples_per_sec_base:.2f} samples/s)")
        print(f"  Saved to: {base_save_dir}")

        results['Base FM'][num_steps] = {
            'time': time_base,
            'samples_per_sec': samples_per_sec_base,
            'save_dir': str(base_save_dir)
        }

        # Generate samples with rectified flow
        print(f"\n{args.reflow_iteration}-Rectified Flow:")
        samples_rectified, time_rectified = generate_samples(
            fm_rectified, args.num_samples, num_steps, args.batch_size, device,
            use_cfg=args.use_cfg, cfg_scale=args.cfg_scale
        )
        samples_per_sec_rectified = args.num_samples / time_rectified

        # Save samples
        rectified_save_dir = save_dir / f"rectified_{args.reflow_iteration}_steps{num_steps}"
        save_samples(samples_rectified, rectified_save_dir)
        print(f"  Time: {time_rectified:.2f}s ({samples_per_sec_rectified:.2f} samples/s)")
        print(f"  Saved to: {rectified_save_dir}")

        results[f'{args.reflow_iteration}-Rectified Flow'][num_steps] = {
            'time': time_rectified,
            'samples_per_sec': samples_per_sec_rectified,
            'save_dir': str(rectified_save_dir)
        }

        # Compute speedup
        speedup = time_base / time_rectified
        print(f"\n  Speedup: {speedup:.2f}x")

    # Save results
    results_file = save_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*80}")
    print(f"Results saved to {results_file}")
    print(f"{'='*80}\n")

    # Plot results
    plot_results(results, args.step_counts, save_dir)

    # Print FID measurement commands
    print("\n" + "="*80)
    print("To measure FID scores, run the following commands:")
    print("="*80)
    for num_steps in args.step_counts:
        base_dir = results['Base FM'][num_steps]['save_dir']
        rectified_dir = results[f'{args.reflow_iteration}-Rectified Flow'][num_steps]['save_dir']
        print(f"\n# {num_steps} steps:")
        print(f"python fid/measure_fid.py data/afhq/eval {base_dir}")
        print(f"python fid/measure_fid.py data/afhq/eval {rectified_dir}")
    print("\n" + "="*80)


def plot_results(results, step_counts, save_dir):
    """Plot sampling time comparison"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    base_times = [results['Base FM'][s]['time'] for s in step_counts]
    rectified_key = list(results.keys())[1]  # Get rectified flow key
    rectified_times = [results[rectified_key][s]['time'] for s in step_counts]

    x = np.arange(len(step_counts))
    width = 0.35

    ax.bar(x - width/2, base_times, width, label='Base FM')
    ax.bar(x + width/2, rectified_times, width, label=rectified_key)

    ax.set_xlabel('Number of ODE Steps')
    ax.set_ylabel('Sampling Time (seconds)')
    ax.set_title('Sampling Speed Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(step_counts)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add speedup annotations
    for i, (base_t, rect_t) in enumerate(zip(base_times, rectified_times)):
        speedup = base_t / rect_t
        ax.text(i, max(base_t, rect_t) + 0.5, f'{speedup:.2f}x',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plot_path = save_dir / "sampling_time_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Rectified Flow with different step counts")
    parser.add_argument("--base_ckpt_path", type=str, required=True, help="Path to base FM checkpoint")
    parser.add_argument("--rectified_ckpt_path", type=str, required=True, help="Path to rectified flow checkpoint")
    parser.add_argument("--save_dir", type=str, default="results/evaluation", help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for generation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--step_counts", type=int, nargs='+', default=[5, 10, 20],
                        help="List of step counts to evaluate")
    parser.add_argument("--use_cfg", action="store_true", default=True, help="Use classifier-free guidance")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG guidance scale")
    parser.add_argument("--reflow_iteration", type=int, default=1, help="Reflow iteration number")

    args = parser.parse_args()
    main(args)
