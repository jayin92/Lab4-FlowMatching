"""
Test script to verify reflow dataset was generated correctly.

Usage:
    python task3_rectified_flow/test_reflow_dataset.py --reflow_data_path data/afhq_reflow --use_cfg
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from reflow_dataset import ReflowDataset


def test_reflow_dataset(reflow_data_path, use_cfg=True):
    """Test if reflow dataset is properly generated"""

    print("="*80)
    print("Testing Reflow Dataset")
    print("="*80)

    dataset_path = Path(reflow_data_path)

    # Check if directory exists
    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset directory not found at {dataset_path}")
        return False

    print(f"✓ Dataset directory exists: {dataset_path}")

    # Check metadata
    metadata_path = dataset_path / "metadata.json"
    if not metadata_path.exists():
        print(f"❌ WARNING: metadata.json not found")
    else:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Metadata found:")
        print(f"  - Num samples: {metadata.get('num_samples', 'N/A')}")
        print(f"  - Inference steps: {metadata.get('num_inference_steps', 'N/A')}")
        print(f"  - Use CFG: {metadata.get('use_cfg', 'N/A')}")
        print(f"  - Image resolution: {metadata.get('image_resolution', 'N/A')}")

    # Count files
    x0_files = sorted(dataset_path.glob("*_x0.pt"))
    z1_files = sorted(dataset_path.glob("*_z1.pt"))
    label_files = sorted(dataset_path.glob("*_label.pt"))

    print(f"\n✓ File counts:")
    print(f"  - x0 files: {len(x0_files)}")
    print(f"  - z1 files: {len(z1_files)}")
    if use_cfg:
        print(f"  - label files: {len(label_files)}")

    if len(x0_files) != len(z1_files):
        print(f"❌ ERROR: Mismatch between x0 and z1 file counts!")
        return False

    if use_cfg and len(label_files) != len(x0_files):
        print(f"❌ ERROR: Mismatch between label and data file counts!")
        return False

    # Load dataset
    try:
        dataset = ReflowDataset(reflow_data_path, use_cfg=use_cfg)
        print(f"\n✓ Dataset loaded successfully with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ ERROR loading dataset: {e}")
        return False

    # Test loading samples
    print("\nTesting sample loading...")
    try:
        if use_cfg:
            x_0, z_1, label = dataset[0]
            print(f"✓ Successfully loaded sample 0:")
            print(f"  - x_0 shape: {x_0.shape}, dtype: {x_0.dtype}")
            print(f"  - z_1 shape: {z_1.shape}, dtype: {z_1.dtype}")
            print(f"  - label: {label.item()}, dtype: {label.dtype}")
        else:
            x_0, z_1 = dataset[0]
            print(f"✓ Successfully loaded sample 0:")
            print(f"  - x_0 shape: {x_0.shape}, dtype: {x_0.dtype}")
            print(f"  - z_1 shape: {z_1.shape}, dtype: {z_1.dtype}")
    except Exception as e:
        print(f"❌ ERROR loading sample: {e}")
        return False

    # Check data statistics
    print("\nChecking data statistics...")

    # x_0 should be Gaussian noise
    x_0_mean = x_0.mean().item()
    x_0_std = x_0.std().item()
    print(f"✓ x_0 statistics:")
    print(f"  - Mean: {x_0_mean:.4f} (expected ~0.0)")
    print(f"  - Std: {x_0_std:.4f} (expected ~1.0)")

    if abs(x_0_mean) > 0.5:
        print(f"  ⚠ WARNING: x_0 mean deviates from 0")
    if abs(x_0_std - 1.0) > 0.5:
        print(f"  ⚠ WARNING: x_0 std deviates from 1")

    # z_1 should be in normalized image range [-1, 1]
    z_1_min = z_1.min().item()
    z_1_max = z_1.max().item()
    z_1_mean = z_1.mean().item()
    print(f"✓ z_1 statistics:")
    print(f"  - Min: {z_1_min:.4f} (expected >= -1.0)")
    print(f"  - Max: {z_1_max:.4f} (expected <= 1.0)")
    print(f"  - Mean: {z_1_mean:.4f}")

    if z_1_min < -1.5 or z_1_max > 1.5:
        print(f"  ⚠ WARNING: z_1 values outside expected range [-1, 1]")

    # Check label range (should be 1, 2, or 3 for AFHQ)
    if use_cfg:
        if label.item() not in [1, 2, 3]:
            print(f"  ⚠ WARNING: Label {label.item()} not in expected range [1, 2, 3]")

    # Test dataloader
    print("\nTesting DataLoader...")
    try:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=4, shuffle=True, num_workers=2
        )
        if use_cfg:
            x_0_batch, z_1_batch, label_batch = next(iter(dataloader))
            print(f"✓ DataLoader works:")
            print(f"  - x_0 batch shape: {x_0_batch.shape}")
            print(f"  - z_1 batch shape: {z_1_batch.shape}")
            print(f"  - label batch shape: {label_batch.shape}")
            print(f"  - labels: {label_batch.tolist()}")
        else:
            x_0_batch, z_1_batch = next(iter(dataloader))
            print(f"✓ DataLoader works:")
            print(f"  - x_0 batch shape: {x_0_batch.shape}")
            print(f"  - z_1 batch shape: {z_1_batch.shape}")
    except Exception as e:
        print(f"❌ ERROR with DataLoader: {e}")
        return False

    # Visualize samples
    print("\nGenerating visualization...")
    try:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        for i in range(4):
            if use_cfg:
                x_0_vis, z_1_vis, label_vis = dataset[i]
            else:
                x_0_vis, z_1_vis = dataset[i]
                label_vis = None

            # x_0 (noise) - show first channel
            axes[0, i].imshow(x_0_vis[0].cpu().numpy(), cmap='gray')
            axes[0, i].set_title(f'x_0 (noise) {i}')
            axes[0, i].axis('off')

            # z_1 (generated image)
            z_1_vis_img = (z_1_vis.cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
            axes[1, i].imshow(z_1_vis_img)
            title = f'z_1 (generated) {i}'
            if label_vis is not None:
                title += f'\nClass: {label_vis.item()}'
            axes[1, i].set_title(title)
            axes[1, i].axis('off')

        plt.tight_layout()
        viz_path = dataset_path / "reflow_dataset_visualization.png"
        plt.savefig(viz_path, dpi=100, bbox_inches='tight')
        print(f"✓ Visualization saved to: {viz_path}")
        plt.close()
    except Exception as e:
        print(f"⚠ WARNING: Could not generate visualization: {e}")

    print("\n" + "="*80)
    print("✓ All tests passed! Reflow dataset is ready for training.")
    print("="*80)
    print(f"\nYou can now train rectified flow with:")
    print(f"python task3_rectified_flow/train_rectified.py \\")
    print(f"  --reflow_data_path {reflow_data_path} \\")
    if use_cfg:
        print(f"  --use_cfg \\")
    print(f"  --reflow_iteration 1")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test reflow dataset")
    parser.add_argument("--reflow_data_path", type=str, required=True,
                        help="Path to reflow dataset directory")
    parser.add_argument("--use_cfg", action="store_true",
                        help="Dataset includes class labels")
    args = parser.parse_args()

    success = test_reflow_dataset(args.reflow_data_path, args.use_cfg)

    if not success:
        print("\n❌ Some tests failed. Please check the errors above.")
        exit(1)
    else:
        exit(0)
