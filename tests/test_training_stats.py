#!/usr/bin/env python
"""
Test script to verify that training statistics enhancements work correctly.
This test validates the syntax and basic functionality of the logging improvements.
"""
import os
import sys
import py_compile

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_module_compilation():
    """Test that modified modules compile successfully"""
    print("Testing module compilation...")
    
    # Get the base directory (parent of tests directory)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    modules_to_test = [
        'histocc/seq2seq_mixer_engine.py',
        'histocc/seq2seq_engine.py',
        'histocc/trainer.py',
    ]
    
    for module in modules_to_test:
        try:
            module_path = os.path.join(base_dir, module)
            py_compile.compile(module_path, doraise=True)
            print(f"✓ {module} compiles successfully")
        except py_compile.PyCompileError as e:
            print(f"✗ {module} compilation error: {e}")
            return False
        except Exception as e:
            print(f"✗ {module} unexpected error: {e}")
            return False
    
    return True


def test_training_functions_exist():
    """Test that key training functions are defined"""
    print("\nTesting function definitions...")
    
    # Get the base directory (parent of tests directory)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Test seq2seq_mixer_engine.py
    with open(os.path.join(base_dir, 'histocc/seq2seq_mixer_engine.py'), 'r') as f:
        content = f.read()
    
    required_functions = [
        'def train_one_epoch',
        'def evaluate',
        'def train',
        '_save_model_checkpoint',
    ]
    
    for func in required_functions:
        if func in content:
            print(f"✓ {func} found in seq2seq_mixer_engine.py")
        else:
            print(f"✗ {func} NOT found in seq2seq_mixer_engine.py")
            return False
    
    # Test seq2seq_engine.py
    with open(os.path.join(base_dir, 'histocc/seq2seq_engine.py'), 'r') as f:
        content = f.read()
    
    required_functions = [
        'def train_one_epoch',
        'def evaluate',
        'def train',
    ]
    
    for func in required_functions:
        if func in content:
            print(f"✓ {func} found in seq2seq_engine.py")
        else:
            print(f"✗ {func} NOT found in seq2seq_engine.py")
            return False
    
    # Test trainer.py
    with open(os.path.join(base_dir, 'histocc/trainer.py'), 'r') as f:
        content = f.read()
    
    required_functions = [
        'def train_epoch',
        'def eval_model',
        'def trainer_loop',
        'def trainer_loop_simple',
        'def run_eval',
        'def run_eval_simple',
    ]
    
    for func in required_functions:
        if func in content:
            print(f"✓ {func} found in trainer.py")
        else:
            print(f"✗ {func} NOT found in trainer.py")
            return False
    
    return True


def test_stats_logging_improvements():
    """Test that enhanced logging features are present"""
    print("\nTesting logging improvements...")
    
    # Get the base directory (parent of tests directory)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Test seq2seq_mixer_engine.py
    with open(os.path.join(base_dir, 'histocc/seq2seq_mixer_engine.py'), 'r') as f:
        content = f.read()
    
    improvements = [
        ('Learning Rate logging', 'current_lr'),
        ('ETA calculation', 'eta_str'),
        ('GPU memory stats', 'torch.cuda.max_memory_allocated'),
        ('Samples per second', 'samples_per_sec'),
        ('Evaluation results header', 'EVALUATION RESULTS'),
        ('Epoch progress', 'Starting Epoch'),
        ('Training complete message', 'TRAINING COMPLETE'),
    ]
    
    for name, pattern in improvements:
        if pattern in content:
            print(f"✓ {name} improvement found")
        else:
            print(f"✗ {name} improvement NOT found")
            return False
    
    # Test seq2seq_engine.py
    with open(os.path.join(base_dir, 'histocc/seq2seq_engine.py'), 'r') as f:
        content = f.read()
    
    improvements = [
        ('Learning Rate logging', 'current_lr'),
        ('ETA calculation', 'eta_str'),
        ('GPU memory stats', 'torch.cuda.max_memory_allocated'),
        ('Evaluation results header', 'EVALUATION RESULTS'),
    ]
    
    for name, pattern in improvements:
        if pattern in content:
            print(f"✓ {name} improvement found in seq2seq_engine.py")
        else:
            print(f"✗ {name} improvement NOT found in seq2seq_engine.py")
            return False
    
    # Test trainer.py
    with open(os.path.join(base_dir, 'histocc/trainer.py'), 'r') as f:
        content = f.read()
    
    improvements = [
        ('Enhanced epoch header', "'='*80"),
        ('Evaluation results header', 'EVALUATION RESULTS'),
        ('Average loss in batch output', 'avg:'),
    ]
    
    for name, pattern in improvements:
        if pattern in content:
            print(f"✓ {name} improvement found in trainer.py")
        else:
            print(f"✗ {name} improvement NOT found in trainer.py")
            return False
    
    return True


def test_backward_compatibility():
    """Test that function signatures remain compatible"""
    print("\nTesting backward compatibility...")
    
    # Get the base directory (parent of tests directory)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check that train_one_epoch signature in seq2seq_mixer_engine.py still has key parameters
    with open(os.path.join(base_dir, 'histocc/seq2seq_mixer_engine.py'), 'r') as f:
        content = f.read()
    
    # Find train_one_epoch function
    if 'def train_one_epoch(' in content:
        # Extract function definition
        start = content.find('def train_one_epoch(')
        # Find the closing of parameters
        depth = 0
        in_params = False
        for i in range(start, len(content)):
            if content[i] == '(':
                depth += 1
                in_params = True
            elif content[i] == ')':
                depth -= 1
                if depth == 0 and in_params:
                    func_def = content[start:i+1]
                    break
        
        # Check for key parameters
        required_params = [
            'model',
            'data_loader',
            'loss_fn',
            'optimizer',
            'device',
            'scheduler',
            'current_step',
        ]
        
        for param in required_params:
            if param in func_def:
                print(f"✓ Parameter '{param}' present in train_one_epoch")
            else:
                print(f"✗ Parameter '{param}' MISSING from train_one_epoch")
                return False
    else:
        print("✗ train_one_epoch function not found")
        return False
    
    return True


def main():
    """Run all tests"""
    print("="*80)
    print("Testing Training Statistics Enhancements")
    print("="*80)
    
    tests = [
        ("Module Compilation", test_module_compilation),
        ("Function Definitions", test_training_functions_exist),
        ("Stats Logging Improvements", test_stats_logging_improvements),
        ("Backward Compatibility", test_backward_compatibility),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        try:
            if not test_func():
                print(f"\n✗ {test_name} FAILED")
                all_passed = False
            else:
                print(f"\n✓ {test_name} PASSED")
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("All tests PASSED ✓")
        print("="*80)
        return 0
    else:
        print("Some tests FAILED ✗")
        print("="*80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
