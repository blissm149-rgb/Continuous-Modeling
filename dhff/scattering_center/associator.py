"""Module 4: Scattering center association via Hungarian algorithm."""
from __future__ import annotations

import numpy as np
import scipy.optimize

from dhff.core.types import ScatteringCenter


class ScatteringCenterAssociator:
    """Match simulation scattering centers to measurement scattering centers."""

    def __init__(self, max_association_distance_m: float = 0.5):
        self.max_association_distance_m = max_association_distance_m

    def associate(
        self,
        sim_centers: list[ScatteringCenter],
        meas_centers: list[ScatteringCenter],
    ) -> tuple[
        list[tuple[ScatteringCenter, ScatteringCenter]],
        list[ScatteringCenter],
        list[ScatteringCenter],
    ]:
        """Use Hungarian algorithm to match sim and meas centers."""
        if not sim_centers or not meas_centers:
            return [], list(sim_centers), list(meas_centers)

        n_sim = len(sim_centers)
        n_meas = len(meas_centers)

        # Cost matrix: Euclidean distance
        C = np.zeros((n_sim, n_meas))
        for i, sc in enumerate(sim_centers):
            for j, mc in enumerate(meas_centers):
                dist = np.sqrt((sc.x - mc.x)**2 + (sc.y - mc.y)**2)
                C[i, j] = dist

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(C)

        matched = []
        unmatched_sim_idx = set(range(n_sim))
        unmatched_meas_idx = set(range(n_meas))

        for r, c in zip(row_ind, col_ind):
            if C[r, c] <= self.max_association_distance_m:
                matched.append((sim_centers[r], meas_centers[c]))
                unmatched_sim_idx.discard(r)
                unmatched_meas_idx.discard(c)

        unmatched_sim = [sim_centers[i] for i in sorted(unmatched_sim_idx)]
        unmatched_meas = [meas_centers[j] for j in sorted(unmatched_meas_idx)]

        return matched, unmatched_sim, unmatched_meas
