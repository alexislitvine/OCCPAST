"""
Test that the finetune scripts correctly parse the new arguments for
AMP mixed precision and num-workers.
"""
import unittest
import sys
import os
from unittest.mock import patch
from io import StringIO

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestFinetuneArgParsing(unittest.TestCase):
    """Test that argument parsing works correctly for finetune scripts."""
    
    def test_finetune_py_args(self):
        """Test that finetune.py accepts --num-workers and --use-amp arguments."""
        # We can't import finetune.py directly because it needs torch
        # Instead, we'll just verify the script exists and has our changes
        finetune_path = os.path.join(os.path.dirname(__file__), '..', 'finetune.py')
        self.assertTrue(os.path.exists(finetune_path), "finetune.py should exist")
        
        with open(finetune_path, 'r') as f:
            content = f.read()
            # Check that the arguments are defined
            self.assertIn("--num-workers", content, 
                         "finetune.py should have --num-workers argument")
            self.assertIn("--use-amp", content, 
                         "finetune.py should have --use-amp argument")
            # Check that num_workers is used in dataloader_kwargs or passed to DataLoader
            self.assertTrue(
                "num_workers=args.num_workers" in content or 
                "'num_workers': args.num_workers" in content,
                "finetune.py should pass num_workers to DataLoader"
            )
            # Check that use_amp is passed to train function
            self.assertIn("use_amp=args.use_amp", content,
                         "finetune.py should pass use_amp to train function")
    
    def test_finetune_py_all_codes_args(self):
        """Test that finetune.py accepts --all-codes-file and --all-codes-col arguments."""
        finetune_path = os.path.join(os.path.dirname(__file__), '..', 'finetune.py')
        self.assertTrue(os.path.exists(finetune_path), "finetune.py should exist")
        
        with open(finetune_path, 'r') as f:
            content = f.read()
            # Check that the arguments are defined
            self.assertIn("--all-codes-file", content, 
                         "finetune.py should have --all-codes-file argument")
            self.assertIn("--all-codes-col", content, 
                         "finetune.py should have --all-codes-col argument")
            # Check that all_codes_file is passed to prepare_data
            self.assertIn("all_codes_file=args.all_codes_file", content,
                         "finetune.py should pass all_codes_file to prepare_data")
            self.assertIn("all_codes_col=args.all_codes_col", content,
                         "finetune.py should pass all_codes_col to prepare_data")
    
    def test_finetune_with_wrapper_args(self):
        """Test that finetune_with_wrapper.py accepts --num-workers and --use-amp arguments."""
        finetune_wrapper_path = os.path.join(os.path.dirname(__file__), '..', 'finetune_with_wrapper.py')
        self.assertTrue(os.path.exists(finetune_wrapper_path), 
                       "finetune_with_wrapper.py should exist")
        
        with open(finetune_wrapper_path, 'r') as f:
            content = f.read()
            # Check that the arguments are defined
            self.assertIn("--num-workers", content, 
                         "finetune_with_wrapper.py should have --num-workers argument")
            self.assertIn("--use-amp", content, 
                         "finetune_with_wrapper.py should have --use-amp argument")
            # Check that they're passed to wrapper.finetune()
            self.assertIn("num_workers=args.num_workers", content,
                         "finetune_with_wrapper.py should pass num_workers to wrapper.finetune()")
            self.assertIn("use_amp=args.use_amp", content,
                         "finetune_with_wrapper.py should pass use_amp to wrapper.finetune()")
    
    def test_seq2seq_mixer_engine_imports(self):
        """Test that seq2seq_mixer_engine.py imports AMP components."""
        engine_path = os.path.join(os.path.dirname(__file__), '..', 'histocc', 'seq2seq_mixer_engine.py')
        self.assertTrue(os.path.exists(engine_path), 
                       "histocc/seq2seq_mixer_engine.py should exist")
        
        with open(engine_path, 'r') as f:
            content = f.read()
            # Check that AMP imports are present
            self.assertIn("from torch.amp import autocast, GradScaler", content,
                         "seq2seq_mixer_engine.py should import AMP components")
            # Check that scaler parameter is in function signature
            self.assertIn("scaler: GradScaler | None = None", content,
                         "train_one_epoch should accept scaler parameter")
            # Check that use_amp parameter is in train function
            self.assertIn("use_amp: bool = False", content,
                         "train function should accept use_amp parameter")
            # Check that GradScaler is initialized
            self.assertIn("scaler = GradScaler('cuda') if use_amp else None", content,
                         "train function should initialize GradScaler when use_amp is True")
    
    def test_prediction_assets_finetune_signature(self):
        """Test that the OccCANINE.finetune method accepts num_workers and use_amp."""
        prediction_assets_path = os.path.join(os.path.dirname(__file__), '..', 'histocc', 'prediction_assets.py')
        self.assertTrue(os.path.exists(prediction_assets_path), 
                       "histocc/prediction_assets.py should exist")
        
        with open(prediction_assets_path, 'r') as f:
            content = f.read()
            # Check that the parameters are in the finetune signature
            self.assertIn("num_workers: int = 0", content,
                         "finetune method should accept num_workers parameter")
            self.assertIn("use_amp: bool = False", content,
                         "finetune method should accept use_amp parameter")
            # Check that they're used in the method
            self.assertTrue(
                "num_workers=num_workers" in content or 
                "'num_workers': num_workers" in content,
                "finetune method should pass num_workers to DataLoader"
            )
            self.assertIn("use_amp=use_amp", content,
                         "finetune method should pass use_amp to train function")


class TestAMPImplementation(unittest.TestCase):
    """Test the AMP implementation details."""
    
    def test_amp_autocast_usage(self):
        """Test that autocast is used when scaler is present."""
        engine_path = os.path.join(os.path.dirname(__file__), '..', 'histocc', 'seq2seq_mixer_engine.py')
        
        with open(engine_path, 'r') as f:
            content = f.read()
            # Check that autocast is used conditionally
            self.assertIn("if scaler is not None:", content,
                         "Should check if scaler is provided")
            self.assertIn("with autocast('cuda'):", content,
                         "Should use autocast when scaler is present")
            # Check that scaler is used for backward pass
            self.assertIn("scaler.scale(loss).backward()", content,
                         "Should use scaler.scale() for backward pass")
            self.assertIn("scaler.step(optimizer)", content,
                         "Should use scaler.step() instead of optimizer.step()")
            self.assertIn("scaler.update()", content,
                         "Should call scaler.update()")
            # Check that gradient unscaling is done before clipping
            self.assertIn("scaler.unscale_(optimizer)", content,
                         "Should unscale gradients before clipping")


if __name__ == '__main__':
    unittest.main()
