"""
The PointerNet: an attention-based sequence model that maps cores to
IO channels, with an auxiliary head that predicts a preemption bucket
per core.

Two fixes live in this file:

  FIX #1 (IO-conditioned preemption head): the head takes
      cat(decoder_output, io_embedding(selected_io)) rather than just
      decoder_output. Without this, the head can only represent the
      marginal P(preempt | core state), not P(preempt | core, IO) —
      but the right preemption schedule genuinely depends on which IO
      channel a core was routed to.

  FIX #6 (backbone gradient isolation): the preemption head's forward
      pass, in the TRAINING path (`get_log_prob_components`), is fed
      `output.detach()`. This is the asymmetry to notice: the mapping
      head's log-prob is computed from `output` directly (gradients
      flow into encoder/decoder/attn), while the preemption head's
      log-prob is computed from `output.detach()` (gradients stop
      there, only io_embed + preemption_head get updated by the
      preemption loss). The pretrained mapping backbone is protected
      from noisy preemption gradients. `forward()` (used for sampling,
      not training) does NOT detach — there's no gradient to isolate
      there since nothing is being backpropagated through a sampling
      pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import IO_EMBED_DIM, MAX_NUM_IO, NUM_PREEMPTION_BUCKETS


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, p, device):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=p).to(device)

    def forward(self, input, mask):
        lengths = mask.sum(dim=1).cpu()
        packed_inputs = pack_padded_sequence(input, lengths, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.rnn(packed_inputs)
        output, _ = pad_packed_sequence(packed_outputs)
        return output, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, p, device):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=p).to(device)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        output = output.squeeze(0)
        return output, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, device, logit_clipping=True, clip_value=10):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(enc_dim + dec_dim, dec_dim), nn.Tanh()).to(device)
        self.v = nn.Linear(dec_dim, 1, bias=False).to(device)
        self.logit_clipping = logit_clipping
        self.clip_value = clip_value

    def forward(self, decoder_output, encoder_outputs, num_io, mask):
        seq_len = encoder_outputs.shape[0]
        decoder_output = decoder_output.unsqueeze(1).repeat(1, seq_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = self.attn(torch.cat((decoder_output, encoder_outputs), dim=2))
        attention = self.v(energy).squeeze(2)
        if self.logit_clipping:
            attention = self.clip_value * torch.tanh(attention)
        attention = attention.masked_fill(mask == 0, float("-inf"))
        attentions, _ = torch.topk(attention, k=num_io, dim=1)
        return attentions


class PointerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, p, device,
                 num_preemption_buckets=NUM_PREEMPTION_BUCKETS,
                 max_num_io=MAX_NUM_IO, io_embed_dim=IO_EMBED_DIM):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, n_layers, p, device)
        self.decoder = Decoder(input_dim, hidden_dim, n_layers, p, device)
        self.attn = Attention(hidden_dim, hidden_dim, device)
        self.initial_decoder_input = nn.Parameter(torch.zeros(1, input_dim))
        self.device = device
        self.num_preemption_buckets = num_preemption_buckets

        # Fix #1: IO embedding for conditioning the preemption head.
        self.io_embed = nn.Embedding(max_num_io, io_embed_dim).to(device)

        # Preemption head input: cat(decoder_output, io_embedding).
        self.preemption_head = nn.Sequential(
            nn.Linear(hidden_dim + io_embed_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_preemption_buckets),
        ).to(device)

    def _preempt_logits(self, output, io_indices, detach_backbone=True):
        """
        Preemption logits conditional on the selected IO (fix #1).

        detach_backbone controls fix #6: when True, `output` is
        detached before entering the head, so no gradient from the
        preemption loss reaches encoder/decoder/attn — only io_embed
        and preemption_head receive gradient. Callers set this True
        during training (get_log_prob_components) and False during
        sampling (forward), where there's nothing to protect since no
        backward pass happens on a sampling-only trajectory.
        """
        feat = output.detach() if detach_backbone else output
        io_emb = self.io_embed(io_indices)
        return self.preemption_head(torch.cat([feat, io_emb], dim=-1))

    def forward(self, input, num_io, mask, num_samples=1, use_preemption_head=True):
        """
        Autoregressive sampling pass: for each position, sample a core's
        IO assignment, then (if use_preemption_head) sample its
        preemption bucket conditioned on that assignment.

        use_preemption_head=False samples preemptions uniformly at
        random instead of from the head — this is the
        --disable_preemption_head ablation mode, isolating the mapping
        head's contribution from the preemption head's.

        Returns (predicted_mappings, predicted_preemptions, log_probs_sum).
        predicted_preemptions holds BUCKET INDICES, not raw preemption
        values — convert with ga.bucket_indices_to_preemptions.
        """
        batch_size = input.size(1)
        seq_len = input.size(0)
        predicted_mappings = torch.zeros(batch_size * num_samples, seq_len, dtype=torch.long, device=self.device)
        predicted_preemptions = torch.zeros(batch_size * num_samples, seq_len, dtype=torch.long, device=self.device)

        encoder_outputs, (hidden, cell) = self.encoder(input, mask)
        encoder_outputs = encoder_outputs.repeat_interleave(num_samples, dim=1)
        hidden = hidden.repeat_interleave(num_samples, dim=1)
        cell = cell.repeat_interleave(num_samples, dim=1)
        mask_decoding = mask.repeat_interleave(num_samples, dim=0).clone()
        decoder_input = self.initial_decoder_input.repeat(batch_size * num_samples, 1)
        log_probs_sum = 0

        for t in range(seq_len):
            output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)

            # --- Mapping head ---
            logits = self.attn(output, encoder_outputs, num_io, mask_decoding)
            log_probs = F.log_softmax(logits, dim=1)
            selected_indices = torch.multinomial(log_probs.exp(), 1).squeeze(1)
            predicted_mappings[:, t] = selected_indices
            selected_log_probs = log_probs.gather(1, selected_indices.unsqueeze(1)).squeeze(1)
            log_probs_sum = log_probs_sum + selected_log_probs

            # --- Preemption head (IO-conditioned, fix #1) ---
            if use_preemption_head:
                preemption_logits = self._preempt_logits(output, selected_indices, detach_backbone=False)
                preemption_log_probs = F.log_softmax(preemption_logits, dim=1)
                preemption_indices = torch.multinomial(preemption_log_probs.exp(), 1).squeeze(1)
                predicted_preemptions[:, t] = preemption_indices
                preempt_lp = preemption_log_probs.gather(1, preemption_indices.unsqueeze(1)).squeeze(1)
                log_probs_sum = log_probs_sum + preempt_lp
            else:
                rand_indices = torch.randint(
                    0, self.num_preemption_buckets,
                    (predicted_preemptions.size(0),), device=self.device,
                )
                predicted_preemptions[:, t] = rand_indices

            input_expanded = input.repeat_interleave(num_samples, dim=1)
            gather_indices = selected_indices.view(1, -1, 1).expand(1, -1, input.size(2))
            decoder_input = input_expanded.gather(0, gather_indices).squeeze(0)
            mask_decoding.scatter_(1, selected_indices.unsqueeze(1), 0)

        return predicted_mappings, predicted_preemptions, log_probs_sum

    def get_log_prob_components(self, input, mask, num_io, target_mappings, target_preemptions=None):
        """
        Teacher-forced log-probabilities under given target mappings
        (and optionally target preemption bucket indices). This is the
        method REINFORCE calls: it replays a fixed trajectory (from the
        GA population) and asks "what log-prob would the current policy
        assign to this trajectory."

        Fix #6: the preemption log-prob uses output.detach() internally
        (via _preempt_logits(..., detach_backbone=True)), so its
        gradient cannot reach encoder/decoder/attn — only io_embed and
        preemption_head are updated by preemption loss. The mapping
        log-prob has no such detach: its gradient flows through the
        full backbone, as intended.

        Returns (mapping_lp, preemption_lp). preemption_lp is None when
        target_preemptions is None.
        """
        batch_size = input.size(1)
        seq_len = input.size(0)
        encoder_outputs, (hidden, cell) = self.encoder(input, mask)
        decoder_input = self.initial_decoder_input.repeat(batch_size, 1)
        mask_decoding = mask.clone()

        mapping_lp = torch.zeros(batch_size, device=self.device)
        preempt_lp = torch.zeros(batch_size, device=self.device) if target_preemptions is not None else None

        for t in range(seq_len):
            output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)

            # Mapping log-prob: gradients flow into encoder/decoder/attn.
            logits = self.attn(output, encoder_outputs, num_io, mask_decoding)
            log_probs = F.log_softmax(logits, dim=1)
            actions = target_mappings[:, t]
            mapping_lp = mapping_lp + log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Preemption log-prob: gradients flow ONLY into io_embed +
            # preemption_head (fix #6), NOT the backbone above.
            if target_preemptions is not None:
                preempt_logits = self._preempt_logits(output, actions, detach_backbone=True)
                preempt_log_probs = F.log_softmax(preempt_logits, dim=1)
                pa = target_preemptions[:, t]
                preempt_lp = preempt_lp + preempt_log_probs.gather(1, pa.unsqueeze(1)).squeeze(1)

            gather_indices = actions.view(1, -1, 1).expand(1, -1, input.size(2))
            decoder_input = input.gather(0, gather_indices).squeeze(0)
            mask_decoding.scatter_(1, actions.unsqueeze(1), 0)

        return mapping_lp, preempt_lp

    def get_log_prob(self, input, mask, num_io, target_mappings, target_preemptions=None):
        """Backwards-compatible wrapper: returns mapping_lp + preemption_lp
        summed (or just mapping_lp if no preemption targets given)."""
        m_lp, p_lp = self.get_log_prob_components(input, mask, num_io, target_mappings, target_preemptions)
        return m_lp + p_lp if p_lp is not None else m_lp