from __future__ import annotations

import numpy as np
import torch


def align_predictor_log_probs_to_batch(
    *,
    original_sids: np.ndarray,
    updated_sids: np.ndarray,
    updated_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Broadcast updated predictor log-probs back to the original batch order.

    `updated_sids` correspond to the filtered batch that actually ran predictor
    PPO updates. Any original SID that was skipped should keep its old
    predictor log-probs so the downstream actor importance ratio stays neutral.
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
