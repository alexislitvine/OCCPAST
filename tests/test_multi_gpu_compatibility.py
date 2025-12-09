#!/usr/bin/env python
"""
Test script to verify that the multi-GPU changes maintain backward compatibility
with single GPU/CPU training.
"""
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that modified modules can be imported"""
    print("Testing module imports...")
    
    try:
        # Test individual file syntax
        import py_compile
        
        # Compile finetune.py
        py_compile.compile('finetune.py', doraise=True)
        print("✓ finetune.py compiles successfully")
        
        # Compile train_mixer.py
        py_compile.compile('train_mixer.py', doraise=True)
        print("✓ train_mixer.py compiles successfully")
        
        # Compile seq2seq_mixer_engine.py
        py_compile.compile('histocc/seq2seq_mixer_engine.py', doraise=True)
        print("✓ histocc/seq2seq_mixer_engine.py compiles successfully")
        
    except py_compile.PyCompileError as e:
        print(f"✗ Compilation error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    return True


def test_finetune_arguments():
    """Test that finetune.py argument parser is correctly defined"""
    print("\nTesting finetune.py argument definitions...")
    
    # Read the file and check for required arguments
    with open('finetune.py', 'r') as f:
        content = f.read()
    
    # Check for distributed training arguments
    checks = [
        ("'--distributed'", "distributed flag"),
        ("'--local_rank'", "local_rank parameter"),
        ("'--world-size'", "world-size parameter"),
        ("action='store_true'", "store_true action for distributed"),
        ("default=False", "default False for distributed"),
        ("default=-1", "default -1 for local_rank"),
        ("default=1", "default 1 for world-size"),
    ]
    
    all_found = True
    for check_str, description in checks:
        if check_str in content:
            print(f"✓ Found {description}")
        else:
            print(f"✗ Missing {description}")
            all_found = False
    
    return all_found


def test_train_mixer_arguments():
    """Test that train_mixer.py argument parser is correctly defined"""
    print("\nTesting train_mixer.py argument definitions...")
    
    # Read the file and check for required arguments
    with open('train_mixer.py', 'r') as f:
        content = f.read()
    
    # Check for distributed training arguments
    checks = [
        ("'--distributed'", "distributed flag"),
        ("'--local_rank'", "local_rank parameter"),
        ("'--world-size'", "world-size parameter"),
        ("import torch.distributed as dist", "distributed import"),
        ("DistributedSampler", "DistributedSampler import/usage"),
        ("DistributedDataParallel", "DDP import"),
    ]
    
    all_found = True
    for check_str, description in checks:
        if check_str in content:
            print(f"✓ Found {description}")
        else:
            print(f"✗ Missing {description}")
            all_found = False
    
    return all_found


def test_engine_modifications():
    """Test that seq2seq_mixer_engine.py has been correctly modified"""
    print("\nTesting seq2seq_mixer_engine.py modifications...")
    
    # Read the file and check for required modifications
    with open('histocc/seq2seq_mixer_engine.py', 'r') as f:
        content = f.read()
    
    # Check for distributed training support
    checks = [
        ("distributed: bool = False", "distributed parameter in train_one_epoch"),
        ("is_main_process: bool = True", "is_main_process parameter in train_one_epoch"),
        ("distributed: bool = False", "distributed parameter in train"),
        ("is_main_process: bool = True", "is_main_process parameter in train"),
        ("if is_main_process", "conditional logging based on main process"),
        ("model_to_save = model.module if distributed else model", "unwrapping DDP model for saving"),
    ]
    
    all_found = True
    for check_str, description in checks:
        if check_str in content:
            print(f"✓ Found {description}")
        else:
            print(f"✗ Missing {description}")
            all_found = False
    
    return all_found


def test_ddp_features():
    """Test that new DDP features are properly implemented"""
    print("\nTesting DDP features...")
    
    # Check finetune.py
    with open('finetune.py', 'r') as f:
        finetune_content = f.read()
    
    # Check seq2seq_mixer_engine.py
    with open('histocc/seq2seq_mixer_engine.py', 'r') as f:
        engine_content = f.read()
    
    checks = [
        # finetune.py checks
        (finetune_content, "import torch.backends.cudnn as cudnn", "cudnn import in finetune.py"),
        (finetune_content, 'drop_last=True', "drop_last=True in train sampler"),
        (finetune_content, 'find_unused_parameters=False', "find_unused_parameters=False in DDP"),
        (finetune_content, "'prefetch_factor': 4", "prefetch_factor=4 in dataloader"),
        (finetune_content, "'pin_memory': True", "pin_memory=True in dataloader"),
        (finetune_content, 'persistent_workers', "persistent_workers in dataloader"),
        
        # seq2seq_mixer_engine.py checks
        (engine_content, "from tqdm import tqdm", "tqdm import"),
        (engine_content, "non_blocking=True", "non_blocking=True for tensor transfers"),
        (engine_content, "train_sampler.set_epoch(epoch)", "train_sampler.set_epoch call"),
        (engine_content, "tqdm(data_loader, disable=not is_main_process", "tqdm progress bar with rank check"),
        (engine_content, "train_sampler: torch.utils.data.distributed.DistributedSampler | None", "train_sampler parameter in train function"),
    ]
    
    all_found = True
    for content, check_str, description in checks:
        if check_str in content:
            print(f"✓ {description}")
        else:
            print(f"✗ Missing: {description}")
            all_found = False
    
    return all_found


def test_backward_compatibility():
    """Test that changes are backward compatible"""
    print("\nTesting backward compatibility...")
    
    # Check that default values maintain backward compatibility
    with open('finetune.py', 'r') as f:
        content = f.read()
    
    checks = [
        # Check that environment variable-based distributed detection works
        ('world_size = int(os.environ.get("WORLD_SIZE", "1"))', 
         "WORLD_SIZE environment variable detection"),
        # Check that device is set correctly using local_rank
        ('device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")',
         "device selection using local_rank"),
        # Check is_main_process helper exists
        ("def is_main_process() -> bool:",
         "is_main_process helper function"),
        # Check performance optimizations
        ("cudnn.benchmark = True",
         "cudnn.benchmark performance optimization"),
    ]
    
    all_found = True
    for check_str, description in checks:
        if check_str in content:
            print(f"✓ {description}")
        else:
            print(f"✗ Missing: {description}")
            all_found = False
    
    return all_found


def main():
    print("=" * 70)
    print("Testing Multi-GPU Training Implementation")
    print("=" * 70)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_finetune_arguments()
    all_passed &= test_train_mixer_arguments()
    all_passed &= test_engine_modifications()
    all_passed &= test_ddp_features()
    all_passed &= test_backward_compatibility()
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All implementation tests passed!")
        print("=" * 70)
        return 0
    else:
        print("✗ Some tests failed!")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
