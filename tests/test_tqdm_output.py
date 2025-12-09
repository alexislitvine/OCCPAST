#!/usr/bin/env python
"""
Test script to verify that tqdm.write is used instead of print
to avoid progress bar output interference.
"""
import os
import sys
import re

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_seq2seq_mixer_engine_uses_tqdm_write():
    """Test that seq2seq_mixer_engine.py uses tqdm.write for logging during training"""
    print("Testing tqdm.write usage in seq2seq_mixer_engine.py...")
    
    # Get the project root directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(test_dir)
    engine_path = os.path.join(project_root, 'histocc', 'seq2seq_mixer_engine.py')
    
    with open(engine_path, 'r') as f:
        content = f.read()
    
    # Check that tqdm is imported
    if "from tqdm import tqdm" not in content:
        print("✗ tqdm import not found")
        return False
    print("✓ tqdm is imported")
    
    # Split content into lines for analysis
    lines = content.split('\n')
    
    # Find the train_one_epoch and evaluate functions
    in_train_one_epoch = False
    in_evaluate = False
    tqdm_iterator_found = False
    issues = []
    
    for i, line in enumerate(lines, 1):
        # Track which function we're in
        if line.startswith('def train_one_epoch('):
            in_train_one_epoch = True
            in_evaluate = False
            tqdm_iterator_found = False
        elif line.startswith('def evaluate('):
            in_evaluate = True
            in_train_one_epoch = False
        elif line.startswith('def ') and not line.strip().startswith('def _'):
            in_train_one_epoch = False
            in_evaluate = False
            tqdm_iterator_found = False
        
        # Detect when tqdm iterator is created
        if in_train_one_epoch and 'tqdm(data_loader' in line:
            tqdm_iterator_found = True
        
        # Check for problematic print statements after tqdm iterator creation
        if (in_train_one_epoch and tqdm_iterator_found) or in_evaluate:
            # Look for print( that is not in a comment
            if re.search(r'^\s*[^#]*\bprint\s*\(', line):
                issues.append(f"Line {i}: Found print() statement that should use tqdm.write(): {line.strip()}")
    
    # Use regex patterns to check for tqdm.write usage (more robust than exact string matching)
    # Note: patterns are flexible to handle multi-line strings and minor format changes
    tqdm_write_patterns = [
        (r'tqdm\.write\(f\'\[Epoch.*Batch', "Batch logging"),
        (r'tqdm\.write\(f\'  Eval Batch', "Evaluation batch logging"),
        (r'tqdm\.write\(f\'  GPU Memory', "GPU memory logging"),
        (r'tqdm\.write\(f\'EVALUATION RESULTS', "Evaluation results header"),
    ]
    
    for pattern, description in tqdm_write_patterns:
        if re.search(pattern, content):
            print(f"✓ {description} uses tqdm.write")
        else:
            issues.append(f"{description} should use tqdm.write")
    
    if issues:
        print("\n✗ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("✓ All logging within tqdm context uses tqdm.write")
    return True


def main():
    print("=" * 70)
    print("Testing tqdm Output Management")
    print("=" * 70)
    
    result = test_seq2seq_mixer_engine_uses_tqdm_write()
    
    print("\n" + "=" * 70)
    if result:
        print("✓ All tests passed!")
        print("=" * 70)
        return 0
    else:
        print("✗ Some tests failed!")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
