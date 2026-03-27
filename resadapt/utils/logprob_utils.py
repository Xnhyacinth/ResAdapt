from __future__ import annotations

import numpy as np
import torch


def align_allocator_log_probs_to_batch(
    *,
    original_sids: np.ndarray,
    updated_sids: np.ndarray,
    updated_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Broadcasts updated allocator log-probabilities back to the original batch order.

    `updated_sids` corresponds to the filtered batch that actually ran allocator
    PPO updates. Any original sequence ID (SID) that was skipped will retain its old
    allocator log-probabilities so the downstream actor importance ratio stays neutral.
    
    Args:
        original_sids (np.ndarray): Array of sequence IDs for the original batch.
        updated_sids (np.ndarray): Array of sequence IDs for the updated batch.
        updated_log_probs (torch.Tensor): Tensor containing the new log-probabilities.
        old_log_probs (torch.Tensor): Tensor containing the old log-probabilities.
        
    Returns:
        torch.Tensor: A tensor of log-probabilities aligned to the original batch order.
    """

    if updated_log_probs.shape[0] != len(updated_sids):
        raise ValueError(
            "`updated_log_probs` and `updated_sids` must have the same length, "
            f"got {updated_log_probs.shape[0]} and {len(updated_sids)}."
        )

    sid_to_row = {
        sid: updated_log_probs[idx]
        for idx, sid in enumerate(updated_sids.tolist())
    }

    aligned_rows = []
    for row_idx, sid in enumerate(original_sids.tolist()):
        aligned_rows.append(sid_to_row.get(sid, old_log_probs[row_idx]))

    return torch.stack(aligned_rows, dim=0)
