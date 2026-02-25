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


def test_decode_constrain_to_valid_blocks_trie():
    class _DummyTrieModel(torch.nn.Module):
        def __init__(self, vocab_size=16):
            super().__init__()
            self.vocab_size = vocab_size

        def encode(self, descr, input_attention_mask):
            bsz = descr.size(0)
            return torch.zeros(bsz, 1, 4, device=descr.device)

        def decode(self, memory, target, target_mask, target_padding_mask):
            bsz = target.size(0)
            logits = torch.full((bsz, target.size(1), self.vocab_size), -10.0, device=target.device)
            logits[:, -1, 7] = 10.0
            logits[:, -1, 3] = 9.0
            logits[:, -1, 4] = 8.0
            return logits

    model = _DummyTrieModel()
    descr = torch.ones(1, 3, dtype=torch.long)
    attn = torch.ones(1, 3, dtype=torch.long)
    valid_blocks = [[3, 3, 3, 3, 3, 3, 3, 3], [4, 4, 4, 4, 4, 4, 4, 4], [3, 4, 3, 4, 3, 4, 3, 4]]

    seq, _ = greedy_decode(
        model=model,
        descr=descr,
        input_attention_mask=attn,
        device=torch.device("cpu"),
        max_len=18,
        start_symbol=BOS_IDX,
        pad_idx=PAD_IDX,
        block_size=8,
        max_num_codes=2,
        constrain_to_valid_blocks=True,
        valid_block_token_ids=valid_blocks,
    )

    block1 = seq[0, 1:9].tolist()
    assert block1 in valid_blocks


def test_decode_constrain_to_valid_blocks_and_no_pad_inside_second_block():
    class _DummyBlock2Model(torch.nn.Module):
        def __init__(self, vocab_size=16):
            super().__init__()
            self.vocab_size = vocab_size

        def encode(self, descr, input_attention_mask):
            bsz = descr.size(0)
            return torch.zeros(bsz, 1, 4, device=descr.device)

        def decode(self, memory, target, target_mask, target_padding_mask):
            bsz = target.size(0)
            seq_len = target.size(1)
            pos = seq_len - 1
            logits = torch.full((bsz, seq_len, self.vocab_size), -12.0, device=target.device)
            if pos == 8:
                logits[:, -1, 5] = 7.0
            elif 9 <= pos <= 15:
                logits[:, -1, PAD_IDX] = 9.0
                logits[:, -1, 6] = 8.0
            else:
                logits[:, -1, 5] = 8.0
            return logits

    model = _DummyBlock2Model()
    descr = torch.ones(1, 3, dtype=torch.long)
    attn = torch.ones(1, 3, dtype=torch.long)
    valid_blocks = [[5, 6, 6, 6, 6, 6, 6, 6], [5, 5, 5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6, 6, 6]]

    seq, _ = greedy_decode(
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
        constrain_to_valid_blocks=True,
        valid_block_token_ids=valid_blocks,
    )

    block2 = seq[0, 9:17].tolist()
    assert PAD_IDX not in block2
    assert block2 in valid_blocks
