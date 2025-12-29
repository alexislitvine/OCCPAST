import torch

from torch import nn
from torch.utils.data import DataLoader

from histocc.seq2seq_mixer_engine import evaluate
from histocc.loss import LossMixer, BlockOrderInvariantLoss
from histocc.formatter import PAD_IDX


class DummySeq2SeqModel(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, num_classes_flat: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_classes_flat = num_classes_flat

    def forward(self, input_ids, attention_mask, target, target_mask, target_padding_mask):
        batch_size = input_ids.size(0)
        out_seq2seq = torch.zeros(batch_size, self.seq_len, self.vocab_size)
        out_linear = torch.zeros(batch_size, self.num_classes_flat)
        return out_seq2seq, out_linear


def test_evaluate_handles_gold_num_codes():
    batch_size = 2
    block_size = 4
    nb_blocks = 2
    vocab_size = 16
    seq_len = block_size * nb_blocks + 1
    target_len = block_size * nb_blocks + 2

    dataset = [
        {
            "input_ids": torch.zeros(8, dtype=torch.long),
            "attention_mask": torch.ones(8, dtype=torch.long),
            "targets_seq2seq": torch.full((target_len,), PAD_IDX, dtype=torch.long),
            "targets_linear": torch.zeros(3, dtype=torch.float),
            "gold_num_codes": torch.tensor(1, dtype=torch.long),
        }
        for _ in range(batch_size)
    ]
    data_loader = DataLoader(dataset, batch_size=batch_size)

    model = DummySeq2SeqModel(vocab_size=vocab_size, seq_len=seq_len, num_classes_flat=3)
    loss_fn_seq2seq = BlockOrderInvariantLoss(
        pad_idx=PAD_IDX,
        nb_blocks=nb_blocks,
        block_size=block_size,
    )
    loss_fn = LossMixer(
        loss_fn_seq2seq=loss_fn_seq2seq,
        loss_fn_linear=nn.BCEWithLogitsLoss(),
        seq2seq_weight=0.5,
    )

    evaluate(
        model=model,
        data_loader=data_loader,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
        log_interval=1,
    )
