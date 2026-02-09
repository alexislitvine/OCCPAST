import numpy as np
import pandas as pd
import torch

from histocc.formatter import construct_general_purpose_formatter, PAD_IDX, BOS_IDX
from histocc.seq2seq_mixer_engine import _flat_accuracy_from_seq2seq
from histocc.utils.decoder import greedy_decode
from histocc.target_cleaning import clean_target_value, get_gold_num_codes_from_values


def _make_formatter():
    return construct_general_purpose_formatter(block_size=8, target_cols=["pst2_1", "pst2_2"])


def _seq_from_codes(formatter, code1, code2):
    row = pd.Series({"pst2_1": code1, "pst2_2": code2})
    return torch.tensor(formatter.transform_label(row), dtype=torch.long)


def test_clean_target_value_and_gold_num_codes():
    assert clean_target_value("nan") is None
    assert clean_target_value(" NaN ") is None
    assert clean_target_value(" ") is None
    assert get_gold_num_codes_from_values(["1,2,3", None]) == 1
    assert get_gold_num_codes_from_values(["1,2,3", "nan"]) == 1


def test_flat_acc_cases():
    formatter = _make_formatter()
    inv_key = {
        "1,2,3,4,5": 0,
        "6,7,8": 1,
        "9,1,2": 2,
    }
    use_sep = True

    g_single = _seq_from_codes(formatter, "1,2,3,4,5,0,0,0", None)
    p_single = g_single.clone()

    g_double = _seq_from_codes(formatter, "1,2,3,4,5,0,0,0", "6,7,8,0,0,0,0,0")
    p_double = g_double.clone()

    p_swapped = _seq_from_codes(formatter, "6,7,8,0,0,0,0,0", "1,2,3,4,5,0,0,0")
    p_wrong2 = _seq_from_codes(formatter, "1,2,3,4,5,0,0,0", "9,1,2,0,0,0,0,0")
    p_missing2 = _seq_from_codes(formatter, "1,2,3,4,5,0,0,0", None)

    preds = torch.stack([p_single, p_double, p_swapped, p_wrong2, p_missing2, p_single])
    golds = torch.stack([g_single, g_double, g_double, g_double, g_double, g_single])

    acc = _flat_accuracy_from_seq2seq(preds, golds, formatter, inv_key, use_sep)
    # correct: exact single, exact double, swapped double, final single-vs-none
    assert np.isclose(acc, (4 / 6) * 100.0)


class _DummyDecodeModel(torch.nn.Module):
    def __init__(self, vocab_size=8):
        super().__init__()
        self.vocab_size = vocab_size

    def encode(self, descr, input_attention_mask):
        bsz = descr.size(0)
        return torch.zeros(bsz, 1, 4, device=descr.device)

    def decode(self, memory, target, target_mask, target_padding_mask):
        bsz = target.size(0)
        logits = torch.full((bsz, target.size(1), self.vocab_size), -10.0, device=target.device)
        # make PAD highest everywhere, token 3 second best
        logits[:, -1, PAD_IDX] = 10.0
        logits[:, -1, 3] = 9.0
        return logits


def test_decode_constrain_pad_within_block():
    model = _DummyDecodeModel()
    descr = torch.ones(1, 3, dtype=torch.long)
    attn = torch.ones(1, 3, dtype=torch.long)

    seq_free, _ = greedy_decode(
        model=model,
        descr=descr,
        input_attention_mask=attn,
        device=torch.device("cpu"),
        max_len=18,
        start_symbol=BOS_IDX,
        pad_idx=PAD_IDX,
        block_size=8,
        max_num_codes=2,
        disallow_pad_inside_block=False,
    )
    seq_constrained, _ = greedy_decode(
        model=model,
        descr=descr,
        input_attention_mask=attn,
        device=torch.device("cpu"),
        max_len=18,
        start_symbol=BOS_IDX,
        pad_idx=PAD_IDX,
        block_size=8,
        max_num_codes=2,
        disallow_pad_inside_block=True,
    )

    # position 2 corresponds to inside first block; unconstrained picks PAD, constrained cannot.
    assert int(seq_free[0, 2].item()) == PAD_IDX
    assert int(seq_constrained[0, 2].item()) != PAD_IDX
