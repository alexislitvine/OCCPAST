import torch

from torch import nn, Tensor

from .masking import generate_square_subsequent_mask
from ..model_assets import Seq2SeqOccCANINE, Seq2SeqMixerOccCANINE, CANINEOccupationClassifier
from ..formatter.hisco import BlockyHISCOFormatter
from .trie import TrieNode, build_trie # For full search


def greedy_decode(
        model: Seq2SeqOccCANINE,
        descr: Tensor,
        input_attention_mask: Tensor,
        device: torch.device,
        max_len: int,
        start_symbol: int,
        pad_idx: int | None = None,
        block_size: int | None = None,
        max_num_codes: int | None = None,
        disallow_pad_inside_block: bool = False,
        disallow_zero_at_block_start: bool = False,
        zero_idx: int | None = None,
        ) -> tuple[Tensor, Tensor]:
    memory = model.encode(descr, input_attention_mask)
    batch_size = descr.size(0)

    # Initialize sequence by placing BoS symbol.
    seq = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
    prob_seq = torch.ones(batch_size, 1).fill_(1.0).type(torch.long).to(device)
    code_region_len = None
    if (disallow_pad_inside_block or disallow_zero_at_block_start) and block_size is not None and max_num_codes is not None:
        code_region_len = block_size * max_num_codes
    finished_code_region = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        target_mask = generate_square_subsequent_mask(seq.shape[1], device).type(torch.bool) # TODO do we need cast?

        out = model.decode(
            memory=memory,
            target=seq,
            target_mask=target_mask,
            target_padding_mask=None,
            )[:, -1:, :] # Only use the prediction for the next token in seq

        if disallow_pad_inside_block and code_region_len is not None and seq.shape[1] <= code_region_len:
            pos = seq.shape[1] - 1
            pos_in_block = pos % block_size
            if pos_in_block != 0:
                out[..., pad_idx] = torch.finfo(out.dtype).min
        if disallow_zero_at_block_start and zero_idx is not None and code_region_len is not None and seq.shape[1] <= code_region_len:
            pos = seq.shape[1] - 1
            pos_in_block = pos % block_size
            if pos_in_block == 0:
                out[..., zero_idx] = torch.finfo(out.dtype).min

        next_token = torch.argmax(out, dim=2).detach()
        next_prob = torch.max(nn.functional.softmax(out, dim=2), dim=2)[0].detach()

        if disallow_pad_inside_block and code_region_len is not None and pad_idx is not None:
            pos = seq.shape[1] - 1
            pos_in_block = pos % block_size
            in_code_region = pos >= 0 and pos < code_region_len
            pad_at_boundary = (next_token.squeeze(1) == pad_idx) & (pos_in_block == 0) & in_code_region
            if pad_at_boundary.any():
                finished_code_region = finished_code_region | pad_at_boundary
            if finished_code_region.any() and pos < code_region_len:
                next_token = torch.where(
                    finished_code_region.unsqueeze(1),
                    torch.full_like(next_token, pad_idx),
                    next_token,
                )

        # Extend sequence by adding prediction of next token.
        seq = torch.cat([seq, next_token], dim=1)
        prob_seq = torch.cat([prob_seq, next_prob], dim=1)

    return seq, prob_seq


def mixer_greedy_decode(
        model: Seq2SeqMixerOccCANINE,
        descr: Tensor,
        input_attention_mask: Tensor,
        device: torch.device,
        max_len: int,
        start_symbol: int,
        linear_topk: int = 5,
        pad_idx: int | None = None,
        block_size: int | None = None,
        max_num_codes: int | None = None,
        disallow_pad_inside_block: bool = False,
        disallow_zero_at_block_start: bool = False,
        zero_idx: int | None = None,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    memory, pooled_memory = model.encode(descr, input_attention_mask)
    batch_size = descr.size(0)

    # Linear output
    out_linear = model.linear_decoder(pooled_memory)
    out_linear = model.linear_decoder_drop(out_linear)
    prob_linear_topk, linear_topk = torch.sigmoid(out_linear).topk(linear_topk, axis=1)
    prob_linear_topk, linear_topk = prob_linear_topk.detach(), linear_topk.detach()

    # seq2seq output
    # Initialize sequence by placing BoS symbol.
    seq = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
    prob_seq = torch.ones(batch_size, 1).fill_(1.0).type(torch.long).to(device)
    code_region_len = None
    if (disallow_pad_inside_block or disallow_zero_at_block_start) and block_size is not None and max_num_codes is not None:
        code_region_len = block_size * max_num_codes
    finished_code_region = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        target_mask = generate_square_subsequent_mask(seq.shape[1], device).type(torch.bool) # TODO do we need cast?

        out = model.decode(
            memory=memory,
            target=seq,
            target_mask=target_mask,
            target_padding_mask=None,
            )[:, -1:, :] # Only use the prediction for the next token in seq

        if disallow_pad_inside_block and code_region_len is not None and seq.shape[1] <= code_region_len:
            pos = seq.shape[1] - 1
            pos_in_block = pos % block_size
            if pos_in_block != 0:
                out[..., pad_idx] = torch.finfo(out.dtype).min
        if disallow_zero_at_block_start and zero_idx is not None and code_region_len is not None and seq.shape[1] <= code_region_len:
            pos = seq.shape[1] - 1
            pos_in_block = pos % block_size
            if pos_in_block == 0:
                out[..., zero_idx] = torch.finfo(out.dtype).min

        next_token = torch.argmax(out, dim=2).detach()
        next_prob = torch.max(nn.functional.softmax(out, dim=2), dim=2)[0].detach()

        if disallow_pad_inside_block and code_region_len is not None and pad_idx is not None:
            pos = seq.shape[1] - 1
            pos_in_block = pos % block_size
            in_code_region = pos >= 0 and pos < code_region_len
            pad_at_boundary = (next_token.squeeze(1) == pad_idx) & (pos_in_block == 0) & in_code_region
            if pad_at_boundary.any():
                finished_code_region = finished_code_region | pad_at_boundary
            if finished_code_region.any() and pos < code_region_len:
                next_token = torch.where(
                    finished_code_region.unsqueeze(1),
                    torch.full_like(next_token, pad_idx),
                    next_token,
                )

        # Extend sequence by adding prediction of next token.
        seq = torch.cat([seq, next_token], dim=1)
        prob_seq = torch.cat([prob_seq, next_prob], dim=1)

    return seq, prob_seq, linear_topk, prob_linear_topk


def flat_decode_flat_model(
        model: Seq2SeqMixerOccCANINE,
        descr: Tensor,
        input_attention_mask: Tensor,
        ):
    """
    Minimal decoder to handle everything as decoders in same module.
    Flat decoder for decoding based on 'flat' model (v1 of OccCANINE).

    """

    logits = model.forward(descr, input_attention_mask)

    return logits

def flat_decode_mixer(
        model: Seq2SeqMixerOccCANINE,
        descr: Tensor,
        input_attention_mask: Tensor,
        ):
    """
    Minimal decoder used for fast 'flat' decoding of mixed output models.

    """
    _, pooled_memory = model.encode(descr, input_attention_mask)

    # Linear output
    logits = model.linear_decoder(pooled_memory)

    return logits


def greedy_decode_for_training(
        model: Seq2SeqOccCANINE,
        descr: Tensor,
        input_attention_mask: Tensor,
        device: torch.device,
        max_len: int,
        start_symbol: int,
        ) -> tuple[Tensor, Tensor]:
    memory = model.encode(descr, input_attention_mask)
    batch_size = descr.size(0)

    # Initialize sequence by placing BoS symbol.
    seq = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device).detach()
    output_seq = []

    for _ in range(max_len): # we loop all the way to fill in some value at EOS pos
        target_mask = generate_square_subsequent_mask(seq.shape[1], device).type(torch.bool) # TODO do we need cast?

        out = model.decode(
            memory=memory,
            target=seq,
            target_mask=target_mask,
            target_padding_mask=None,
            )[:, -1:, :] # Only use the prediction for the next token in seq

        next_token = torch.argmax(out, dim=2).detach()

        # Extend sequence by adding prediction of next token.
        seq = torch.cat([seq, next_token], dim=1)
        output_seq.append(out)

    output_seq = torch.cat(output_seq, dim=1)

    return output_seq


def full_search_decoder_seq2seq_optimized(
        model: Seq2SeqOccCANINE,
        descr: torch.Tensor,
        input_attention_mask: torch.Tensor,
        device: torch.device,
        codes_list: list[list[int]],
        start_symbol: int,
        ) -> dict:

    memory = model.encode(descr, input_attention_mask)
    batch_size = descr.size(0)

    # Step 1: Build Trie
    trie = build_trie(codes_list)

    # Step 2: Initialize results
    results = torch.empty((batch_size, len(codes_list)), dtype=torch.float, device=device)
    code_indices = {tuple(code): idx for idx, code in enumerate(codes_list)}

    # Step 3: Initialize sequences
    seq = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
    prob_seq = torch.ones(batch_size, 1).fill_(1.0).type(torch.float).to(device)

    # Step 4: Decode using Trie
    stack = [(trie, seq, prob_seq)]

    # n_model_calls = 0

    while stack:
        node, seq, prob_seq = stack.pop()

        if node.codes:
            for code in node.codes:
                code_seq_probs = prob_seq[:, -1]
                results[:, code_indices[tuple(code)]] = code_seq_probs

        for number, child_node in node.children.items():
            which_output = torch.ones(batch_size, 1).fill_(number).type(torch.long).to(device)
            target_mask = generate_square_subsequent_mask(seq.shape[1], device).type(torch.bool)
            out = model.decode(
                memory=memory,
                target=seq,
                target_mask=target_mask,
                target_padding_mask=None,
            )[:, -1:, :]
            # n_model_calls += 1

            next_prob = torch.gather(torch.nn.functional.softmax(out, dim=2), 2, which_output.unsqueeze(2)).squeeze(2)
            new_prob_seq = prob_seq * next_prob
            new_seq = torch.cat([seq, which_output], dim=1)
            stack.append((child_node, new_seq, new_prob_seq))

    return results


def full_search_decoder_mixer_optimized(
        model: Seq2SeqMixerOccCANINE,
        descr: torch.Tensor,
        input_attention_mask: torch.Tensor,
        device: torch.device,
        codes_list: list[list[int]],
        start_symbol: int,
        ) -> dict:
    memory = model.encode(descr, input_attention_mask)
    
    # Ensure memory is a tensor
    if isinstance(memory, tuple):
        memory = memory[0]

    batch_size = descr.size(0)

    # Step 1: Build Trie
    trie = build_trie(codes_list)

    # Step 2: Initialize results
    results = torch.empty((batch_size, len(codes_list)), dtype=torch.float, device=device)
    code_indices = {tuple(code): idx for idx, code in enumerate(codes_list)}

    # Step 3: Initialize sequences
    seq = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
    prob_seq = torch.ones(batch_size, 1).fill_(1.0).type(torch.float).to(device)

    # Step 4: Decode using Trie
    stack = [(trie, seq, prob_seq)]

    # n_model_calls = 0

    while stack:
        node, seq, prob_seq = stack.pop()

        if node.codes:
            for code in node.codes:
                code_seq_probs = prob_seq[:, -1]
                results[:, code_indices[tuple(code)]] = code_seq_probs

        for number, child_node in node.children.items():
            which_output = torch.ones(batch_size, 1).fill_(number).type(torch.long).to(device)
            target_mask = generate_square_subsequent_mask(seq.shape[1], device).type(torch.bool)
            out = model.decode(
                memory=memory,
                target=seq,
                target_mask=target_mask,
                target_padding_mask=None,
            )[:, -1:, :]
            # n_model_calls += 1

            next_prob = torch.gather(torch.nn.functional.softmax(out, dim=2), 2, which_output.unsqueeze(2)).squeeze(2)
            new_prob_seq = prob_seq * next_prob
            new_seq = torch.cat([seq, which_output], dim=1)
            stack.append((child_node, new_seq, new_prob_seq))

    # print(n_model_calls) # 4073 is equal to trie.count_nodes(): Compared to len(codes_list)*5 = 9595

    return results
