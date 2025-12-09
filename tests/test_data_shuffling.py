"""
Test that the finetuning data preparation shuffles the training data.

This is important because if the input data has different types of data in
sequential order (e.g., all category A followed by all category B), not
shuffling could lead to problems during training.
"""
import unittest
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestDataShufflingInFinetune(unittest.TestCase):
    """Test that data shuffling is implemented in finetune scripts."""

    def test_finetune_py_shuffles_training_data(self):
        """Test that finetune.py shuffles training data before saving."""
        finetune_path = os.path.join(os.path.dirname(__file__), '..', 'finetune.py')
        self.assertTrue(os.path.exists(finetune_path), "finetune.py should exist")

        with open(finetune_path, 'r') as f:
            content = f.read()
            # Check that training data is shuffled
            self.assertIn("data_train.sample(frac=1", content,
                         "finetune.py should shuffle training data using sample(frac=1)")
            # Check that a random state is used for reproducibility
            self.assertIn("random_state=42", content,
                         "finetune.py should use random_state for reproducibility")
            # Check that the index is reset after shuffling
            self.assertIn("reset_index(drop=True)", content,
                         "finetune.py should reset index after shuffling")

    def test_utils_io_shuffles_training_data(self):
        """Test that histocc/utils/io.py shuffles training data before saving."""
        io_path = os.path.join(os.path.dirname(__file__), '..', 'histocc', 'utils', 'io.py')
        self.assertTrue(os.path.exists(io_path), "histocc/utils/io.py should exist")

        with open(io_path, 'r') as f:
            content = f.read()
            # Check that training data is shuffled
            self.assertIn("data_train.sample(frac=1", content,
                         "histocc/utils/io.py should shuffle training data using sample(frac=1)")
            # Check that a random state is used for reproducibility
            self.assertIn("random_state=42", content,
                         "histocc/utils/io.py should use random_state for reproducibility")
            # Check that the index is reset after shuffling
            self.assertIn("reset_index(drop=True)", content,
                         "histocc/utils/io.py should reset index after shuffling")


class TestDataShufflingBehavior(unittest.TestCase):
    """Test that data shuffling works correctly."""

    def test_shuffling_actually_changes_order(self):
        """Test that shuffling with sample(frac=1, random_state=42) actually changes order."""
        import pandas as pd

        # Create sequential data (like data from different sources)
        df = pd.DataFrame({
            'category': ['A'] * 50 + ['B'] * 50,
            'value': list(range(100))
        })

        # Shuffle using the same method as in the code
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Verify that the order has changed
        # The first 50 values should no longer all be 'A'
        first_50_original = df['category'][:50].tolist()
        first_50_shuffled = df_shuffled['category'][:50].tolist()

        self.assertNotEqual(first_50_original, first_50_shuffled,
                           "Shuffling should change the order of data")

        # Verify that all data is preserved
        self.assertEqual(len(df), len(df_shuffled),
                        "Shuffling should preserve all rows")
        self.assertEqual(set(df['value']), set(df_shuffled['value']),
                        "Shuffling should preserve all values")

    def test_shuffling_is_reproducible(self):
        """Test that shuffling with fixed random_state is reproducible."""
        import pandas as pd

        df = pd.DataFrame({
            'category': ['A'] * 50 + ['B'] * 50,
            'value': list(range(100))
        })

        # Shuffle twice with the same random_state
        df_shuffled1 = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df_shuffled2 = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Verify that both shuffles produce the same result
        self.assertTrue(df_shuffled1.equals(df_shuffled2),
                       "Shuffling with same random_state should be reproducible")


if __name__ == '__main__':
    unittest.main()
