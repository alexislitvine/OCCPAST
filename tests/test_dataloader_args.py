"""
Test that the finetune scripts correctly parse the new DataLoader arguments for
--prefetch-factor, --pin-memory, and --persistent-workers.
"""
import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestDataLoaderArgParsing(unittest.TestCase):
    """Test that argument parsing works correctly for DataLoader parameters."""
    
    def test_finetune_py_dataloader_args(self):
        """Test that finetune.py accepts DataLoader arguments."""
        finetune_path = os.path.join(os.path.dirname(__file__), '..', 'finetune.py')
        self.assertTrue(os.path.exists(finetune_path), "finetune.py should exist")
        
        with open(finetune_path, 'r') as f:
            content = f.read()
            # Check that the arguments are defined
            self.assertIn("--prefetch-factor", content, 
                         "finetune.py should have --prefetch-factor argument")
            self.assertIn("--pin-memory", content, 
                         "finetune.py should have --pin-memory argument")
            self.assertIn("--persistent-workers", content, 
                         "finetune.py should have --persistent-workers argument")
            
            # Check that prefetch_factor help text is present
            self.assertIn("Number of batches loaded in advance by each worker", content,
                         "finetune.py should have descriptive help for --prefetch-factor")
            
            # Check that pin_memory help text is present
            self.assertIn("Pin memory for faster data transfer to GPU", content,
                         "finetune.py should have descriptive help for --pin-memory")
            
            # Check that persistent_workers help text is present
            self.assertIn("Keep workers alive between epochs", content,
                         "finetune.py should have descriptive help for --persistent-workers")
    
    def test_finetune_with_wrapper_dataloader_args(self):
        """Test that finetune_with_wrapper.py accepts DataLoader arguments."""
        finetune_wrapper_path = os.path.join(os.path.dirname(__file__), '..', 'finetune_with_wrapper.py')
        self.assertTrue(os.path.exists(finetune_wrapper_path), 
                       "finetune_with_wrapper.py should exist")
        
        with open(finetune_wrapper_path, 'r') as f:
            content = f.read()
            # Check that the arguments are defined
            self.assertIn("--prefetch-factor", content, 
                         "finetune_with_wrapper.py should have --prefetch-factor argument")
            self.assertIn("--pin-memory", content, 
                         "finetune_with_wrapper.py should have --pin-memory argument")
            self.assertIn("--persistent-workers", content, 
                         "finetune_with_wrapper.py should have --persistent-workers argument")
            
            # Check that they're passed to wrapper.finetune()
            self.assertIn("prefetch_factor=args.prefetch_factor", content,
                         "finetune_with_wrapper.py should pass prefetch_factor to wrapper.finetune()")
            self.assertIn("pin_memory=args.pin_memory", content,
                         "finetune_with_wrapper.py should pass pin_memory to wrapper.finetune()")
            self.assertIn("persistent_workers=args.persistent_workers", content,
                         "finetune_with_wrapper.py should pass persistent_workers to wrapper.finetune()")
    
    def test_prediction_assets_finetune_dataloader_signature(self):
        """Test that the OccCANINE.finetune method accepts DataLoader parameters."""
        prediction_assets_path = os.path.join(os.path.dirname(__file__), '..', 'histocc', 'prediction_assets.py')
        self.assertTrue(os.path.exists(prediction_assets_path), 
                       "histocc/prediction_assets.py should exist")
        
        with open(prediction_assets_path, 'r') as f:
            content = f.read()
            # Check that the parameters are in the finetune signature
            self.assertIn("prefetch_factor: int | None = None", content,
                         "finetune method should accept prefetch_factor parameter")
            self.assertIn("pin_memory: bool = False", content,
                         "finetune method should accept pin_memory parameter")
            self.assertIn("persistent_workers: bool = False", content,
                         "finetune method should accept persistent_workers parameter")
            
            # Check that they're mentioned in the docstring
            self.assertIn("prefetch_factor", content,
                         "finetune method docstring should document prefetch_factor")
            self.assertIn("pin_memory", content,
                         "finetune method docstring should document pin_memory")
            self.assertIn("persistent_workers", content,
                         "finetune method docstring should document persistent_workers")


class TestDataLoaderImplementation(unittest.TestCase):
    """Test the DataLoader implementation details."""
    
    def test_finetune_py_dataloader_kwargs(self):
        """Test that finetune.py creates DataLoader with proper kwargs."""
        finetune_path = os.path.join(os.path.dirname(__file__), '..', 'finetune.py')
        
        with open(finetune_path, 'r') as f:
            content = f.read()
            # Check that dataloader_kwargs is built
            self.assertIn("dataloader_kwargs", content,
                         "Should build dataloader_kwargs dictionary")
            self.assertIn("'pin_memory': args.pin_memory", content,
                         "Should include pin_memory in dataloader_kwargs")
            
            # Check conditional handling for num_workers > 0
            self.assertIn("if args.num_workers > 0:", content,
                         "Should check num_workers before setting worker-specific params")
            self.assertIn("if args.persistent_workers:", content,
                         "Should conditionally set persistent_workers")
            self.assertIn("if args.prefetch_factor is not None:", content,
                         "Should conditionally set prefetch_factor")
            
            # Check that kwargs are unpacked into DataLoader
            self.assertIn("**dataloader_kwargs", content,
                         "Should unpack dataloader_kwargs into DataLoader")
    
    def test_prediction_assets_dataloader_kwargs(self):
        """Test that prediction_assets.py creates DataLoader with proper kwargs."""
        prediction_assets_path = os.path.join(os.path.dirname(__file__), '..', 'histocc', 'prediction_assets.py')
        
        with open(prediction_assets_path, 'r') as f:
            content = f.read()
            # Check that dataloader_kwargs is built
            self.assertIn("dataloader_kwargs", content,
                         "Should build dataloader_kwargs dictionary")
            self.assertIn("'pin_memory': pin_memory", content,
                         "Should include pin_memory in dataloader_kwargs")
            
            # Check conditional handling for num_workers > 0
            self.assertIn("if num_workers > 0:", content,
                         "Should check num_workers before setting worker-specific params")
            self.assertIn("if persistent_workers:", content,
                         "Should conditionally set persistent_workers")
            self.assertIn("if prefetch_factor is not None:", content,
                         "Should conditionally set prefetch_factor")
            
            # Check that kwargs are unpacked into DataLoader
            self.assertIn("**dataloader_kwargs", content,
                         "Should unpack dataloader_kwargs into DataLoader")


if __name__ == '__main__':
    unittest.main()
