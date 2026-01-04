# Language-Specific Metrics

## Overview

The training system now tracks sequence accuracy and token accuracy per language during validation. These metrics are automatically logged to WandB and displayed in the console output during evaluation.

## Features

- **Per-Language Accuracy Tracking**: Separate sequence and token accuracy metrics for each language in the validation dataset
- **WandB Integration**: All language-specific metrics are automatically logged to WandB with keys like `seq_acc_en`, `token_acc_de`, etc.
- **DDP Support**: Fully compatible with distributed data parallel training
- **Backward Compatible**: Works seamlessly with existing training code; if language information is not available, the system continues to work without language-specific metrics

## Implementation Details

### Dataset Changes

The `OccDatasetMixerInMemMultipleFiles` dataset now includes the `lang` field in the batch data:

```python
batch_data = {
    'occ1': input_seq,
    'input_ids': encoded_input_seq['input_ids'].flatten(),
    'attention_mask': encoded_input_seq['attention_mask'].flatten(),
    'targets_seq2seq': torch.tensor(targets_seq2seq, dtype=torch.long),
    'targets_linear': torch.tensor(target_linear, dtype=torch.float),
    'gold_num_codes': torch.tensor(gold_num_codes, dtype=torch.long),
    'lang': lang,  # New field
}
```

### Evaluation Changes

The `evaluate()` function in `seq2seq_mixer_engine.py` now:

1. Tracks accuracy per language using `Averager` instances
2. Returns an additional `lang_metrics` dictionary containing:
   - `seq_acc_{lang}`: Sequence accuracy for each language
   - `token_acc_{lang}`: Token accuracy for each language
   - `count_{lang}`: Number of samples for each language

### Console Output

During evaluation, you'll see output like:

```
================================================================================
EVALUATION RESULTS (Step 5000)
================================================================================
Validation Loss     : 0.234567 (Linear: 0.123456, Seq2Seq: 0.345678)
Training Loss       : 0.245678
Sequence Accuracy   : 89.45%
Token Accuracy      : 94.23%
Flat Accuracy       : 91.56%
Learning Rate       : 2.00e-05
--------------------------------------------------------------------------------
LANGUAGE-SPECIFIC METRICS:
      en: Seq Acc: 91.23% | Token Acc: 95.67% | Count: 1234
      de: Seq Acc: 87.89% | Token Acc: 92.45% | Count: 987
      nl: Seq Acc: 88.45% | Token Acc: 93.21% | Count: 876
================================================================================
```

### WandB Logging

All language-specific metrics are automatically logged to WandB with the following keys:
- `seq_acc_{lang}`: Sequence accuracy for language
- `token_acc_{lang}`: Token accuracy for language  
- `count_{lang}`: Sample count for language

These appear alongside other metrics like `seq_acc`, `token_acc`, `val_loss`, etc.

## DDP Compatibility

The implementation is fully compatible with Distributed Data Parallel (DDP) training:

1. **Metric Computation**: Each process computes language metrics for its data subset
2. **Broadcasting**: In DDP mode, language metrics are broadcast from the main process to all other processes
3. **Consistency**: All processes receive the same metric values after synchronization

The broadcast happens in the same section where other evaluation metrics are broadcast:

```python
# Broadcast language-specific metrics
for lang_key, lang_value in lang_metrics.items():
    tensor = torch.tensor(float(lang_value), device=device, dtype=torch.float32)
    ddp_broadcast(tensor, f"metric:{lang_key}", current_step, device)
```

## Backward Compatibility

The implementation maintains full backward compatibility:

1. **Optional Language Field**: The `lang` field is optional in batch data. If not present, language-specific metrics simply won't be computed.
2. **Empty Dict Handling**: When no language data is available, an empty dictionary is returned and safely merged with other metrics.
3. **Existing Training Code**: No changes required to existing training scripts or workflows.

## Usage

No changes are needed to your training scripts. Language-specific metrics are automatically computed and logged when:

1. The dataset includes a `lang` field in the batch data
2. The evaluation function is called during training

For datasets that use `OccDatasetMixerInMemMultipleFiles`, language metrics are automatically enabled since the dataset now includes the language field.

## Performance Considerations

- **Minimal Overhead**: Language-specific metrics add minimal computational overhead as they reuse already-computed accuracy values
- **Per-Sample Computation**: For each sample in a batch, we compute a lightweight per-sample accuracy to attribute it to the correct language
- **Memory Efficient**: Uses `Averager` instances to track running averages without storing all individual values

## Testing

The implementation includes tests to verify:
1. Language metrics are correctly computed and returned
2. Backward compatibility when `lang` field is not present
3. Proper unpacking of the evaluate function's return values

Run tests with:
```bash
python -c "from tests.test_eval_gold_num_codes import test_evaluate_handles_gold_num_codes; test_evaluate_handles_gold_num_codes()"
```
