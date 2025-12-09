"""
Test that the train_mixer.py script correctly parses and uses the
--save-each-epoch argument.
"""
import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestSaveEachEpochArgParsing(unittest.TestCase):
    """Test that argument parsing works correctly for --save-each-epoch flag."""
    
    def test_train_mixer_has_save_each_epoch_arg(self):
        """Test that train_mixer.py has --save-each-epoch argument."""
        train_mixer_path = os.path.join(os.path.dirname(__file__), '..', 'train_mixer.py')
        self.assertTrue(os.path.exists(train_mixer_path), "train_mixer.py should exist")
        
        with open(train_mixer_path, 'r') as f:
            content = f.read()
            # Check that the argument is defined
            self.assertIn("--save-each-epoch", content, 
                         "train_mixer.py should have --save-each-epoch argument")
            self.assertIn("action='store_true'", content,
                         "--save-each-epoch should be a boolean flag")
            # Check that it's passed to the train function
            self.assertIn("save_each_epoch=args.save_each_epoch", content,
                         "train_mixer.py should pass save_each_epoch to train function")
    
    def test_seq2seq_mixer_engine_accepts_save_each_epoch(self):
        """Test that seq2seq_mixer_engine.py accepts save_each_epoch parameter."""
        engine_path = os.path.join(os.path.dirname(__file__), '..', 'histocc', 'seq2seq_mixer_engine.py')
        self.assertTrue(os.path.exists(engine_path), 
                       "histocc/seq2seq_mixer_engine.py should exist")
        
        with open(engine_path, 'r') as f:
            content = f.read()
            # Check that save_each_epoch parameter is in train function signature
            self.assertIn("save_each_epoch: bool = False", content,
                         "train function should accept save_each_epoch parameter")
            # Check that it's passed to train_one_epoch
            self.assertIn("save_each_epoch=save_each_epoch", content,
                         "train function should pass save_each_epoch to train_one_epoch")
    
    def test_epoch_based_saving_logic(self):
        """Test that epoch-based saving logic is implemented."""
        engine_path = os.path.join(os.path.dirname(__file__), '..', 'histocc', 'seq2seq_mixer_engine.py')
        
        with open(engine_path, 'r') as f:
            content = f.read()
            # Check that saving happens at end of epoch when flag is set
            self.assertIn("if save_each_epoch and is_main_process:", content,
                         "Should check save_each_epoch flag before saving")
            # Check that step-based saving is disabled when save_each_epoch is True
            self.assertIn("and not save_each_epoch", content,
                         "Step-based saving should be disabled when save_each_epoch is True")
            # Check that checkpoint function is called
            self.assertIn("_save_model_checkpoint", content,
                         "Should call _save_model_checkpoint function")
    
    def test_save_each_epoch_in_train_one_epoch_signature(self):
        """Test that train_one_epoch accepts save_each_epoch parameter."""
        engine_path = os.path.join(os.path.dirname(__file__), '..', 'histocc', 'seq2seq_mixer_engine.py')
        
        with open(engine_path, 'r') as f:
            content = f.read()
            # Find the train_one_epoch function
            self.assertIn("def train_one_epoch(", content,
                         "train_one_epoch function should exist")
            # The parameter should be in the signature
            lines = content.split('\n')
            in_train_one_epoch = False
            found_save_each_epoch_param = False
            
            for line in lines:
                if 'def train_one_epoch(' in line:
                    in_train_one_epoch = True
                if in_train_one_epoch and 'save_each_epoch: bool = False' in line:
                    found_save_each_epoch_param = True
                    break
                # End of function signature
                if in_train_one_epoch and ') -> int:' in line:
                    break
            
            self.assertTrue(found_save_each_epoch_param,
                           "train_one_epoch should have save_each_epoch parameter")


if __name__ == '__main__':
    unittest.main()
