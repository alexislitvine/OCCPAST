"""
Test that warmup percentage is correctly implemented in training scripts.
"""
import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestWarmupPercentage(unittest.TestCase):
    """Test that warmup percentage is correctly implemented."""
    
    def test_train_py_warmup_pct(self):
        """Test that train.py uses warmup-pct instead of warmup-steps."""
        train_path = os.path.join(os.path.dirname(__file__), '..', 'train.py')
        self.assertTrue(os.path.exists(train_path), "train.py should exist")
        
        with open(train_path, 'r') as f:
            content = f.read()
            # Check that warmup-pct argument exists
            self.assertIn("--warmup-pct", content, 
                         "train.py should have --warmup-pct argument")
            # Check that default is 0.05 (5%)
            self.assertIn("default=0.05", content,
                         "train.py should have default warmup-pct of 0.05")
            # Check that warmup steps are calculated from percentage
            self.assertIn("int(total_steps * args.warmup_pct)", content,
                         "train.py should calculate warmup steps from percentage")
            # Check that old warmup-steps argument is removed
            self.assertNotIn("--warmup-steps", content,
                           "train.py should not have --warmup-steps argument")
    
    def test_train_mixer_py_warmup_pct(self):
        """Test that train_mixer.py uses warmup-pct instead of warmup-steps."""
        train_mixer_path = os.path.join(os.path.dirname(__file__), '..', 'train_mixer.py')
        self.assertTrue(os.path.exists(train_mixer_path), "train_mixer.py should exist")
        
        with open(train_mixer_path, 'r') as f:
            content = f.read()
            self.assertIn("--warmup-pct", content, 
                         "train_mixer.py should have --warmup-pct argument")
            self.assertIn("default=0.05", content,
                         "train_mixer.py should have default warmup-pct of 0.05")
            self.assertIn("int(total_steps * args.warmup_pct)", content,
                         "train_mixer.py should calculate warmup steps from percentage")
            self.assertNotIn("--warmup-steps", content,
                           "train_mixer.py should not have --warmup-steps argument")
    
    def test_train_v2_py_warmup_pct(self):
        """Test that train_v2.py uses warmup-pct instead of warmup-steps."""
        train_v2_path = os.path.join(os.path.dirname(__file__), '..', 'train_v2.py')
        self.assertTrue(os.path.exists(train_v2_path), "train_v2.py should exist")
        
        with open(train_v2_path, 'r') as f:
            content = f.read()
            self.assertIn("--warmup-pct", content, 
                         "train_v2.py should have --warmup-pct argument")
            self.assertIn("default=0.05", content,
                         "train_v2.py should have default warmup-pct of 0.05")
            self.assertIn("int(total_steps * args.warmup_pct)", content,
                         "train_v2.py should calculate warmup steps from percentage")
            self.assertNotIn("--warmup-steps", content,
                           "train_v2.py should not have --warmup-steps argument")
    
    def test_finetune_py_warmup_pct(self):
        """Test that finetune.py uses warmup-pct instead of warmup-steps."""
        finetune_path = os.path.join(os.path.dirname(__file__), '..', 'finetune.py')
        self.assertTrue(os.path.exists(finetune_path), "finetune.py should exist")
        
        with open(finetune_path, 'r') as f:
            content = f.read()
            self.assertIn("--warmup-pct", content, 
                         "finetune.py should have --warmup-pct argument")
            self.assertIn("default=0.05", content,
                         "finetune.py should have default warmup-pct of 0.05")
            self.assertIn("int(total_steps * args.warmup_pct)", content,
                         "finetune.py should calculate warmup steps from percentage")
            self.assertNotIn("--warmup-steps", content,
                           "finetune.py should not have --warmup-steps argument")
    
    def test_finetune_with_wrapper_warmup_pct(self):
        """Test that finetune_with_wrapper.py uses warmup-pct."""
        wrapper_path = os.path.join(os.path.dirname(__file__), '..', 'finetune_with_wrapper.py')
        self.assertTrue(os.path.exists(wrapper_path), "finetune_with_wrapper.py should exist")
        
        with open(wrapper_path, 'r') as f:
            content = f.read()
            self.assertIn("--warmup-pct", content, 
                         "finetune_with_wrapper.py should have --warmup-pct argument")
            self.assertIn("warmup_pct=args.warmup_pct", content,
                         "finetune_with_wrapper.py should pass warmup_pct to wrapper")
            self.assertNotIn("--warmup-steps", content,
                           "finetune_with_wrapper.py should not have --warmup-steps argument")
    
    def test_prediction_assets_warmup_pct(self):
        """Test that prediction_assets.py finetune method uses warmup_pct."""
        assets_path = os.path.join(os.path.dirname(__file__), '..', 'histocc', 'prediction_assets.py')
        self.assertTrue(os.path.exists(assets_path), "histocc/prediction_assets.py should exist")
        
        with open(assets_path, 'r') as f:
            content = f.read()
            self.assertIn("warmup_pct: float = 0.05", content,
                         "finetune method should accept warmup_pct parameter with default 0.05")
            self.assertIn("int(total_steps * warmup_pct)", content,
                         "finetune method should calculate warmup steps from percentage")
    
    def test_seq2seq_wrapper_warmup_pct(self):
        """Test that seq2seq_wrapper.py uses warmup percentage."""
        wrapper_path = os.path.join(os.path.dirname(__file__), '..', 'histocc', 'seq2seq_wrapper.py')
        self.assertTrue(os.path.exists(wrapper_path), "histocc/seq2seq_wrapper.py should exist")
        
        with open(wrapper_path, 'r') as f:
            content = f.read()
            self.assertIn("int(total_steps * 0.05)", content,
                         "seq2seq_wrapper.py should calculate warmup steps from 5% of total")


class TestTF32Support(unittest.TestCase):
    """Test that TF32 support is enabled in all training scripts."""
    
    def test_train_py_tf32(self):
        """Test that train.py enables TF32."""
        train_path = os.path.join(os.path.dirname(__file__), '..', 'train.py')
        
        with open(train_path, 'r') as f:
            content = f.read()
            self.assertIn("torch.backends.cuda.matmul.allow_tf32 = True", content,
                         "train.py should enable TF32 for matmul")
            self.assertIn("torch.backends.cudnn.allow_tf32 = True", content,
                         "train.py should enable TF32 for cudnn")
    
    def test_train_mixer_py_tf32(self):
        """Test that train_mixer.py enables TF32."""
        train_mixer_path = os.path.join(os.path.dirname(__file__), '..', 'train_mixer.py')
        
        with open(train_mixer_path, 'r') as f:
            content = f.read()
            self.assertIn("torch.backends.cuda.matmul.allow_tf32 = True", content,
                         "train_mixer.py should enable TF32 for matmul")
            self.assertIn("torch.backends.cudnn.allow_tf32 = True", content,
                         "train_mixer.py should enable TF32 for cudnn")
    
    def test_train_v2_py_tf32(self):
        """Test that train_v2.py enables TF32."""
        train_v2_path = os.path.join(os.path.dirname(__file__), '..', 'train_v2.py')
        
        with open(train_v2_path, 'r') as f:
            content = f.read()
            self.assertIn("torch.backends.cuda.matmul.allow_tf32 = True", content,
                         "train_v2.py should enable TF32 for matmul")
            self.assertIn("torch.backends.cudnn.allow_tf32 = True", content,
                         "train_v2.py should enable TF32 for cudnn")
    
    def test_finetune_py_tf32(self):
        """Test that finetune.py enables TF32."""
        finetune_path = os.path.join(os.path.dirname(__file__), '..', 'finetune.py')
        
        with open(finetune_path, 'r') as f:
            content = f.read()
            self.assertIn("torch.backends.cuda.matmul.allow_tf32 = True", content,
                         "finetune.py should enable TF32 for matmul")
            self.assertIn("torch.backends.cudnn.allow_tf32 = True", content,
                         "finetune.py should enable TF32 for cudnn")
    
    def test_prediction_assets_tf32(self):
        """Test that prediction_assets.py finetune method enables TF32."""
        assets_path = os.path.join(os.path.dirname(__file__), '..', 'histocc', 'prediction_assets.py')
        
        with open(assets_path, 'r') as f:
            content = f.read()
            self.assertIn("torch.backends.cuda.matmul.allow_tf32 = True", content,
                         "prediction_assets.py should enable TF32 for matmul")
            self.assertIn("torch.backends.cudnn.allow_tf32 = True", content,
                         "prediction_assets.py should enable TF32 for cudnn")
    
    def test_seq2seq_wrapper_tf32(self):
        """Test that seq2seq_wrapper.py enables TF32."""
        wrapper_path = os.path.join(os.path.dirname(__file__), '..', 'histocc', 'seq2seq_wrapper.py')
        
        with open(wrapper_path, 'r') as f:
            content = f.read()
            self.assertIn("torch.backends.cuda.matmul.allow_tf32 = True", content,
                         "seq2seq_wrapper.py should enable TF32 for matmul")
            self.assertIn("torch.backends.cudnn.allow_tf32 = True", content,
                         "seq2seq_wrapper.py should enable TF32 for cudnn")


if __name__ == '__main__':
    unittest.main()
