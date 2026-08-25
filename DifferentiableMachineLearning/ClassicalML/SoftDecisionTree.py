"""Soft / differentiable decision tree (Frosst 2017) re-implemented
as a ``torch.nn.Module``.

A tree of depth ``d`` has ``n_inner = 2^d - 1`` inner nodes and
``n_leaves = 2^d`` leaves. Each inner node carries a learnable linear
filter ``w ∈ R^{n_features}`` and bias ``b ∈ R``; the probability of
routing a sample to its *left* child is

    p_left = sigmoid(temperature * (x . w + b))

and the routing distribution is a soft binary tree. The output is
the expected class distribution: a weighted sum of the (learnable)
per-leaf class distributions, with weights equal to the leaf
posterior.

The whole computation is a single backprop-friendly graph.
"""

import torch


class SoftDecisionTree(torch.nn.Module):
    """
    Analogue:
        Frosst & Hinton 2017 'Distilling a Neural Network Into a Soft Decision Tree'
    Differentiable decision tree.

    Parameters
    ----------
    depth : int
        Tree depth; must be >= 1. The number of leaves is ``2 ** depth``.
    n_features : int
        Input feature dimension.
    n_classes : int
        Number of classes.
    temperature : float, default 1.0
        Multiplier on the inner-node logits. Larger values produce
        sharper (closer-to-hard) splits; smaller values produce a
        smoother tree.
    """

    def __init__(
        self,
        depth: int,
        n_features: int,
        n_classes: int,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if n_features <= 0 or n_classes <= 0:
            raise ValueError("n_features and n_classes must be > 0")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.depth = depth
        self.n_features = n_features
        self.n_classes = n_classes
        self.temperature = temperature
        self.n_inner = 2 ** depth - 1
        self.n_leaves = 2 ** depth

        # Inner-node parameters: W [n_inner, n_features], b [n_inner]
        self.W = torch.nn.Parameter(
            torch.empty(self.n_inner, n_features).uniform_(-0.1, 0.1)
        )
        self.b = torch.nn.Parameter(
            torch.zeros(self.n_inner, dtype=torch.float32)
        )
        # Per-leaf class distribution (logits).
        self.leaf_logits = torch.nn.Parameter(
            torch.zeros(self.n_leaves, n_classes, dtype=torch.float32)
        )

    def _leaf_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """For each sample, the probability of reaching each leaf.

        We track ``routing`` over the *current* level only: at every
        step the current routing has exactly one column per inner node
        at that level. After ``depth`` splits the routing has been
        pushed down to the leaves, so the final ``[batch, n_leaves]``
        routing IS the leaf-probability matrix.

        BFS convention: inner node ``i`` (level order) splits the
        routing mass into its two children; BFS indices for level
        ``l`` are ``2^l - 1 .. 2^(l+1) - 2`` (so for depth 2 the inner
        indices are 0, 1, 2 — but we never use the third because it
        is the last inner level whose children are leaves).
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected input with {self.n_features} features, got {x.shape[1]}"
            )
        routing = torch.ones(x.shape[0], 1, dtype=x.dtype)
        for level in range(self.depth):
            level_start = 2 ** level - 1
            level_end = 2 ** (level + 1) - 1
            level_W = self.W[level_start:level_end]
            level_b = self.b[level_start:level_end]
            logits = self.temperature * (x @ level_W.T + level_b)
            p_left = torch.sigmoid(logits)
            p_right = 1.0 - p_left
            left_routing = routing * p_left
            right_routing = routing * p_right
            # The children become the "current level" for the next
            # iteration. For the last iteration, the children are
            # the leaves and the result is the leaf-probability
            # matrix.
            routing = torch.cat([left_routing, right_routing], dim=1)
        return routing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Soft class distribution per sample, shape ``[batch, n_classes]``."""
        leaf_prob = self._leaf_probabilities(x)                  # [B, n_leaves]
        leaf_dist = torch.nn.functional.softmax(self.leaf_logits, dim=-1)  # [L, C]
        return leaf_prob @ leaf_dist

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.forward(x), dim=-1)

    def extra_repr(self) -> str:
        return (
            f"depth={self.depth}, n_features={self.n_features}, "
            f"n_classes={self.n_classes}, temperature={self.temperature}"
        )