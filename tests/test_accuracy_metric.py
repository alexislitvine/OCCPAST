"""
Test cases for the accuracy metric in eval_metrics.py

This test file validates that the accuracy metric correctly handles cases where:
1. Ground truth has fewer codes than predictions
2. Ground truth has more codes than predictions
3. Perfect matches
4. No matches
"""

import unittest
import pandas as pd
import numpy as np
from histocc import OccCANINE, EvalEngine


class TestAccuracyMetric(unittest.TestCase):
    """Test the accuracy calculation in EvalEngine"""

    def setUp(self):
        """Setup a mock model for testing"""
        # Create a minimal OccCANINE instance for testing
        self.model = OccCANINE(skip_load=True, model_type='mix', system='test')
        
        # Override formatter for testing
        from histocc.formatter import construct_general_purpose_formatter
        self.model.formatter = construct_general_purpose_formatter(
            block_size=5,
            target_cols=['test_1', 'test_2', 'test_3'],
        )
        self.model.system = 'test'

    def test_perfect_match_single_code(self):
        """Test accuracy when prediction perfectly matches single ground truth"""
        ground_truth = pd.DataFrame({'test_1': ['12345']})
        predictions = pd.DataFrame({'test_1': ['12345']})
        
        eval_engine = EvalEngine(self.model, ground_truth, predictions, 'test_')
        accuracy = eval_engine.accuracy()
        
        # Perfect match should give 100% accuracy
        assert accuracy == 1.0, f"Expected 1.0, got {accuracy}"

    def test_single_truth_multiple_predictions_with_match(self):
        """Test when ground truth has 1 code and predictions have 3 codes (1 correct)"""
        ground_truth = pd.DataFrame({'test_1': ['12345']})
        predictions = pd.DataFrame({
            'test_1': ['12345'],
            'test_2': ['67890'],
            'test_3': ['11111']
        })
        
        eval_engine = EvalEngine(self.model, ground_truth, predictions, 'test_')
        accuracy = eval_engine.accuracy()
        
        # With the fixed formula: (1/3 + 1/1) / 2 = 0.667
        # The correct answer is in the predictions, so should be high
        assert accuracy > 0.5, f"Expected > 0.5, got {accuracy}"
        assert np.isclose(accuracy, 2/3, atol=0.01), f"Expected ~0.667, got {accuracy}"

    def test_multiple_truth_single_prediction_with_match(self):
        """Test when ground truth has 3 codes and prediction has 1 code (correct)"""
        ground_truth = pd.DataFrame({
            'test_1': ['12345'],
            'test_2': ['67890'],
            'test_3': ['11111']
        })
        predictions = pd.DataFrame({'test_1': ['12345']})
        
        eval_engine = EvalEngine(self.model, ground_truth, predictions, 'test_')
        accuracy = eval_engine.accuracy()
        
        # With the fixed formula: (1/1 + 1/3) / 2 = 0.667
        assert accuracy > 0.5, f"Expected > 0.5, got {accuracy}"
        assert np.isclose(accuracy, 2/3, atol=0.01), f"Expected ~0.667, got {accuracy}"

    def test_perfect_match_multiple_codes(self):
        """Test accuracy when predictions perfectly match multiple ground truth codes"""
        ground_truth = pd.DataFrame({
            'test_1': ['12345'],
            'test_2': ['67890']
        })
        predictions = pd.DataFrame({
            'test_1': ['12345'],
            'test_2': ['67890']
        })
        
        eval_engine = EvalEngine(self.model, ground_truth, predictions, 'test_')
        accuracy = eval_engine.accuracy()
        
        # Perfect match should give 100% accuracy
        assert accuracy == 1.0, f"Expected 1.0, got {accuracy}"

    def test_no_match(self):
        """Test accuracy when no predictions match ground truth"""
        ground_truth = pd.DataFrame({'test_1': ['12345']})
        predictions = pd.DataFrame({'test_1': ['67890']})
        
        eval_engine = EvalEngine(self.model, ground_truth, predictions, 'test_')
        accuracy = eval_engine.accuracy()
        
        # No match should give 0% accuracy
        assert accuracy == 0.0, f"Expected 0.0, got {accuracy}"

    def test_partial_match_multiple_codes(self):
        """Test accuracy with partial match in multiple codes"""
        ground_truth = pd.DataFrame({
            'test_1': ['12345'],
            'test_2': ['67890']
        })
        predictions = pd.DataFrame({
            'test_1': ['12345'],
            'test_2': ['11111']  # Wrong second code
        })
        
        eval_engine = EvalEngine(self.model, ground_truth, predictions, 'test_')
        accuracy = eval_engine.accuracy()
        
        # 1 out of 2 matched: (1/2 + 1/2) / 2 = 0.5
        assert np.isclose(accuracy, 0.5, atol=0.01), f"Expected 0.5, got {accuracy}"

    def test_accuracy_with_multiple_observations(self):
        """Test average accuracy across multiple observations"""
        ground_truth = pd.DataFrame({
            'test_1': ['12345', '11111'],
        })
        predictions = pd.DataFrame({
            'test_1': ['12345', '11111'],  # Both perfect matches
        })
        
        eval_engine = EvalEngine(self.model, ground_truth, predictions, 'test_')
        accuracy = eval_engine.accuracy()
        
        # Both perfect matches should give 100% accuracy
        assert accuracy == 1.0, f"Expected 1.0, got {accuracy}"

    def test_realistic_scenario(self):
        """
        Test a realistic scenario where model provides top-5 predictions
        and ground truth has 1 code (which is in the predictions)
        """
        ground_truth = pd.DataFrame({'test_1': ['12345']})
        predictions = pd.DataFrame({
            'test_1': ['12345'],  # Correct!
            'test_2': ['67890'],
            'test_3': ['11111'],
        })
        
        eval_engine = EvalEngine(self.model, ground_truth, predictions, 'test_')
        
        # Check individual metrics
        precision = eval_engine.precision()
        recall = eval_engine.recall()
        accuracy = eval_engine.accuracy()
        
        # Precision should be 1/3 (1 correct out of 3 predictions)
        assert np.isclose(precision, 1/3, atol=0.01), f"Expected precision ~0.333, got {precision}"
        
        # Recall should be 1.0 (found the 1 ground truth)
        assert recall == 1.0, f"Expected recall 1.0, got {recall}"
        
        # Accuracy should be average: (1/3 + 1.0) / 2 = 2/3
        assert np.isclose(accuracy, 2/3, atol=0.01), f"Expected accuracy ~0.667, got {accuracy}"


if __name__ == '__main__':
    unittest.main()
