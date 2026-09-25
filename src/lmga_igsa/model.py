"""
PointerNetV2: a two-phase, non-recurrent architecture replacing v1's
LSTM pointer-network. See model_v1_backup.py for the retired
architecture (kept for reference/rollback, not imported anywhere).

PHASE 1 (mapping): every core attends to every IO in parallel via
multi-head cross-attention (cores as queries, IOs as keys/values —
this is the standard attention compatibility mechanism, applied to a
fully-connected bipartite structure, so no explicit graph/adjacency is
needed). One IO is hard-sampled per core.

PHASE 2 (preemption), run AFTER phase 1's mapping is fully resolved:
a single-layer GAT (Graph Attention Network) over the CONFLICT graph —
not the mesh-topology graph, not per-IO groupings — where an edge
between two cores exists iff data.check_path_conflict says their
routing paths would contend for a physical link, given the sampled
mapping. This directly mirrors what simulator.py's scheduler actually
reacts to (see data.build_conflict_table's docstring and
tests/test_data.py's cross-IO-conflict regression test — conflicts are
NOT restricted to cores sharing an IO channel). Each core's conflict
embedding is concatenated with its own core embedding and fed to an
MLP for a per-core preemption bucket.

Per-core input is [row, col, patterns, scan] (CORE_INPUT_DIM_V2 = 4),
not just position as in v1 — patterns (workload size) and scan
(scan-chain overhead) are genuine per-core-varying properties in
standard NoC test benchmarks (see data.py's Core class) and materially
affect the optimal preemption choice (see reinforce.py's module
docstring / the architecture discussion this design came from: scan
sets the per-subtask overhead a core pays repeatedly under fine
preemption; patterns sets how much work there is to chop up).

NO GRADIENT DETACHING between phases (a deliberate departure from v1's
fix #6): both mapping_lp and preempt_lp backprop through the shared
core/IO feature projections. v1's detach existed to protect an
ALREADY-PRETRAINED backbone from an UNTRAINED, noisy preemption head
bolted on afterward — that asymmetry doesn't exist here (both phases
train together from the start via one REINFORCE update), and cutting
the gradient path would prevent the shared embeddings from ever
learning what makes a mapping decision lead to good or bad preemption
scheduling outcomes downstream — information only the preemption
loss's gradient carries.

The conflict table itself is NOT computed by this module. Per an
explicit design decision, it is computed once by the caller (run.py),
right after phase 1's mapping is sampled, via
data.build_conflict_table, and passed into BOTH this model's phase 2
AND simulator.simulate_single_mapping as a shared argument — this
keeps forward()/predict_preemption() pure tensor-in/tensor-out (no
Python-level control flow branching on sampled values inside the
model), and avoids computing the same O(N^2) table twice for the same
individual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    CORE_INPUT_DIM_V2, GAT_LEAKY_SLOPE_V2, HIDDEN_DIM_V2, IO_INPUT_DIM_V2,
    NUM_ATTN_HEADS_V2, NUM_PREEMPTION_BUCKETS,
)


class CoreIOAttention(nn.Module):
    """
    Multi-head cross-attention for phase 1: cores are queries, IOs are
    keys/values. Every core attends to every IO (fully connected
    bipartite — no sparsity, so no explicit adjacency is needed).
    """

    def __init__(self, core_dim, io_dim, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim

        self.W_Q = nn.Linear(core_dim, hidden_dim)
        self.W_K = nn.Linear(io_dim, hidden_dim)
        self.W_V = nn.Linear(io_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, core_feats, io_feats):
        """
        core_feats: (B, Nc, core_dim)
        io_feats:   (B, Nio, io_dim)
        Returns:
          mapping_scores: (B, Nc, Nio) — averaged across heads, the
              single distribution used for sampling/log-prob (a
              multi-head layer naturally produces H attention patterns;
              we average them into one, since the mapping decision is
              a single categorical choice per core, not one per head).
          core_ctx: (B, Nc, hidden_dim) — attended core representation,
              available for downstream use (not currently consumed by
              phase 2, which uses raw core features per the design,
              but kept for future flexibility / diagnostics).
        """
        B, Nc, _ = core_feats.shape
        _, Nio, _ = io_feats.shape

        Q = self.W_Q(core_feats).view(B, Nc, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(io_feats).view(B, Nio, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(io_feats).view(B, Nio, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B,H,Nc,Nio)
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, V)  # (B,H,Nc,d)
        ctx = ctx.transpose(1, 2).contiguous().view(B, Nc, self.hidden_dim)
        ctx = self.out_proj(ctx)

        mapping_scores = scores.mean(dim=1)  # (B, Nc, Nio)
        return mapping_scores, ctx


class ConflictGAT(nn.Module):
    """
    Single-layer Graph Attention Network over the conflict graph
    (data.build_conflict_table's output), for phase 2. Each core
    attends only to cores it actually conflicts with, weighted by
    learned attention coefficients (standard GAT):
        e_ij = LeakyReLU(a_l . Wh_i + a_r . Wh_j)
        alpha_ij = softmax_j(e_ij)   (restricted to i's neighbors)
        h_i' = ELU(sum_j alpha_ij * Wh_j)

    A core with no conflicts (isolated in the conflict graph — common
    and expected, not an edge case to avoid) has no neighbors to
    aggregate; its row in the pre-softmax score matrix is entirely
    masked to -inf, which would produce NaN from a naive softmax. This
    is handled explicitly: such cores' output is set to exactly zero
    rather than left undefined.
    """

    def __init__(self, in_dim, out_dim, num_heads, leaky_slope=GAT_LEAKY_SLOPE_V2):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_l = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attn_r = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_l.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_r.unsqueeze(0))
        self.leaky_slope = leaky_slope

    def forward(self, node_feats, adj):
        """
        node_feats: (B, N, in_dim) — raw per-core features, per design
            (this GAT pass is independent of phase 1's core embedding,
            not layered on top of it).
        adj: (B, N, N) boolean/float — adj[b,i,j] truthy iff cores i,j
            conflict (data.build_conflict_table's output, batched).
        Returns: (B, N, out_dim), zero for nodes with no neighbors.
        """
        B, N, _ = node_feats.shape
        Wh = self.W(node_feats).view(B, N, self.num_heads, self.head_dim)

        e_l = torch.einsum("bnhd,hd->bnh", Wh, self.attn_l)
        e_r = torch.einsum("bnhd,hd->bnh", Wh, self.attn_r)
        e = F.leaky_relu(e_l.unsqueeze(2) + e_r.unsqueeze(1), self.leaky_slope)  # (B,N,N,H)

        adj_mask = adj.bool().unsqueeze(-1)  # (B,N,N,1)
        neg_inf = torch.finfo(e.dtype).min
        e_masked = e.masked_fill(~adj_mask, neg_inf)

        has_neighbor = adj.bool().any(dim=2)  # (B,N) — False for isolated cores

        alpha = F.softmax(e_masked, dim=2)     # softmax over neighbor axis j
        alpha = torch.nan_to_num(alpha, nan=0.0)  # isolated rows: all -inf -> NaN -> 0

        Wh_perm = Wh.permute(0, 2, 1, 3)        # (B,H,N,d)
        alpha_perm = alpha.permute(0, 3, 1, 2)  # (B,H,N,N)
        out = torch.matmul(alpha_perm, Wh_perm)  # (B,H,N,d)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.out_dim)

        out = out * has_neighbor.unsqueeze(-1).to(out.dtype)  # explicit zero for isolated cores
        return F.elu(out)


class PointerNetV2(nn.Module):
    def __init__(self, core_in_dim=CORE_INPUT_DIM_V2, io_in_dim=IO_INPUT_DIM_V2,
                 hidden_dim=HIDDEN_DIM_V2, num_preemption_buckets=NUM_PREEMPTION_BUCKETS,
                 num_heads=NUM_ATTN_HEADS_V2, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.num_preemption_buckets = num_preemption_buckets

        self.core_proj = nn.Linear(core_in_dim, hidden_dim).to(self.device)
        self.io_proj = nn.Linear(io_in_dim, hidden_dim).to(self.device)
        self.phase1 = CoreIOAttention(hidden_dim, hidden_dim, hidden_dim, num_heads).to(self.device)
        self.conflict_gat = ConflictGAT(core_in_dim, hidden_dim, num_heads).to(self.device)
        self.preempt_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_preemption_buckets),
        ).to(self.device)

    # ------------------------------------------------------------------
    # Phase 1: mapping
    # ------------------------------------------------------------------
    def sample_mapping(self, core_raw, io_raw):
        """
        core_raw: (B, Nc, core_in_dim) — [row, col, patterns, scan]
        io_raw:   (B, Nio, io_in_dim)
        Returns: (sampled_mapping [B,Nc] long, mapping_lp [B] float,
                  core_h [B,Nc,hidden] — projected core features, for
                  reuse by the caller if useful).
        Pure sampling: no conflict table is built here (that requires
        the caller to have the resolved mapping first — see module
        docstring). Callers needing teacher-forced log-probs on a
        GIVEN mapping (REINFORCE replay) should use
        mapping_log_prob() instead.
        """
        core_h = self.core_proj(core_raw)
        io_h = self.io_proj(io_raw)
        mapping_scores, _ = self.phase1(core_h, io_h)  # (B, Nc, Nio)

        mapping_logprobs = F.log_softmax(mapping_scores, dim=-1)
        dist = torch.distributions.Categorical(logits=mapping_scores)
        sampled_mapping = dist.sample()  # (B, Nc)
        mapping_lp = mapping_logprobs.gather(-1, sampled_mapping.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

        return sampled_mapping, mapping_lp, core_h

    def mapping_log_prob(self, core_raw, io_raw, target_mapping):
        """
        Teacher-forced mapping log-prob under a GIVEN mapping (e.g.
        from a GA individual's genes), for REINFORCE replay.
        target_mapping: (B, Nc) long.
        Returns: mapping_lp (B,) float.
        """
        core_h = self.core_proj(core_raw)
        io_h = self.io_proj(io_raw)
        mapping_scores, _ = self.phase1(core_h, io_h)
        mapping_logprobs = F.log_softmax(mapping_scores, dim=-1)
        mapping_lp = mapping_logprobs.gather(-1, target_mapping.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        return mapping_lp

    # ------------------------------------------------------------------
    # Phase 2: preemption (requires the resolved mapping's conflict table)
    # ------------------------------------------------------------------
    def predict_preemption(self, core_raw, conflict_adj):
        """
        core_raw: (B, Nc, core_in_dim) — same raw features as phase 1
            (conflict GAT operates on raw features independently, per
            design — not on phase 1's projected core_h).
        conflict_adj: (B, Nc, Nc) — from data.build_conflict_table,
            one table per batch element, built by the caller from each
            element's ALREADY-SAMPLED mapping.
        Returns: (sampled_preempt [B,Nc] long bucket indices,
                  preempt_lp [B] float).
        """
        core_h_for_mlp = self.core_proj(core_raw)  # same projection phase 1 uses, for a consistent core representation into the MLP
        conflict_emb = self.conflict_gat(core_raw, conflict_adj)  # (B, Nc, hidden)

        preempt_logits = self.preempt_mlp(torch.cat([core_h_for_mlp, conflict_emb], dim=-1))
        preempt_logprobs = F.log_softmax(preempt_logits, dim=-1)
        dist = torch.distributions.Categorical(logits=preempt_logits)
        sampled_preempt = dist.sample()
        preempt_lp = preempt_logprobs.gather(-1, sampled_preempt.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

        return sampled_preempt, preempt_lp

    def preemption_log_prob(self, core_raw, conflict_adj, target_preempt):
        """
        Teacher-forced preemption log-prob under GIVEN bucket indices
        (e.g. from a GA individual's stored preemptions, converted via
        ga.preemptions_to_bucket_indices), for REINFORCE replay.
        target_preempt: (B, Nc) long bucket indices.
        Returns: preempt_lp (B,) float.
        """
        core_h_for_mlp = self.core_proj(core_raw)
        conflict_emb = self.conflict_gat(core_raw, conflict_adj)
        preempt_logits = self.preempt_mlp(torch.cat([core_h_for_mlp, conflict_emb], dim=-1))
        preempt_logprobs = F.log_softmax(preempt_logits, dim=-1)
        preempt_lp = preempt_logprobs.gather(-1, target_preempt.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        return preempt_lp

    # ------------------------------------------------------------------
    # Convenience: full teacher-forced pass, mirroring v1's
    # get_log_prob_components(...) contract for reinforce.py.
    # ------------------------------------------------------------------
    def get_log_prob_components(self, core_raw, io_raw, target_mapping, conflict_adj=None,
                                 target_preempt=None):
        """
        mapping_lp is always computed. preempt_lp is computed only if
        BOTH conflict_adj and target_preempt are given (mirrors v1's
        "preemption target optional" contract) — conflict_adj is
        REQUIRED alongside target_preempt here, unlike v1, since phase
        2 cannot run without a resolved mapping's conflict table; the
        caller (reinforce.py) is responsible for building conflict_adj
        via data.build_conflict_table from target_mapping before
        calling this.
        Returns (mapping_lp, preempt_lp); preempt_lp is None if either
        conflict_adj or target_preempt is omitted.
        """
        mapping_lp = self.mapping_log_prob(core_raw, io_raw, target_mapping)

        preempt_lp = None
        if conflict_adj is not None and target_preempt is not None:
            preempt_lp = self.preemption_log_prob(core_raw, conflict_adj, target_preempt)

        return mapping_lp, preempt_lp