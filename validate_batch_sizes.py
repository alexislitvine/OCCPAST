#!/usr/bin/env python3
"""
Helper script to validate and fix batch sizes for distributed training.

This script checks if batch sizes are divisible by the world_size (number of GPUs)
and suggests corrected values if needed.
"""
import argparse
import sys


def validate_and_fix_batch_sizes(batch_sizes, world_size, round_up=True):
    """
    Validate batch sizes and suggest fixes if needed.
    
    Args:
        batch_sizes: List of batch sizes to validate
        world_size: Number of GPUs (world size for distributed training)
        round_up: If True, round up non-divisible sizes; otherwise round down
        
    Returns:
        Tuple of (is_valid, fixed_batch_sizes, messages)
    """
    is_valid = True
    fixed_batch_sizes = []
    messages = []
    
    messages.append(f"Validating batch sizes for world_size={world_size}:")
    
    for bs in batch_sizes:
        if bs % world_size == 0:
            fixed_batch_sizes.append(bs)
            messages.append(f"  {bs}: ✓ Valid (distributes {bs // world_size} samples per GPU)")
        else:
            is_valid = False
            remainder = bs % world_size
            
            if round_up:
                fixed_bs = ((bs // world_size) + 1) * world_size
            else:
                fixed_bs = (bs // world_size) * world_size
            
            fixed_batch_sizes.append(fixed_bs)
            messages.append(f"  {bs}: ✗ INVALID (remainder={remainder})")
            messages.append(f"    → Suggested fix: {fixed_bs} ({fixed_bs // world_size} samples per GPU)")
    
    return is_valid, fixed_batch_sizes, messages


def main():
    parser = argparse.ArgumentParser(
        description="Validate batch sizes for distributed training"
    )
    parser.add_argument(
        'batch_sizes',
        type=int,
        nargs='+',
        help='Batch sizes to validate'
    )
    parser.add_argument(
        '--world-size',
        type=int,
        default=8,
        help='Number of GPUs (default: 8)'
    )
    parser.add_argument(
        '--round-down',
        action='store_true',
        help='Round down instead of up for invalid batch sizes'
    )
    
    args = parser.parse_args()
    
    is_valid, fixed_batch_sizes, messages = validate_and_fix_batch_sizes(
        args.batch_sizes,
        args.world_size,
        round_up=not args.round_down
    )
    
    # Print all messages
    for msg in messages:
        print(msg)
    
    # Print summary
    print("\n" + "="*60)
    if is_valid:
        print("✓ All batch sizes are valid!")
    else:
        print("✗ Some batch sizes need to be fixed.")
        print(f"\nOriginal: {' '.join(map(str, args.batch_sizes))}")
        print(f"Fixed:    {' '.join(map(str, fixed_batch_sizes))}")
        print("\nUpdate your SLURM script with the fixed values.")
    print("="*60)
    
    # Exit with error code if invalid
    sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()
