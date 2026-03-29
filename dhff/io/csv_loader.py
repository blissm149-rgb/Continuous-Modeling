"""CSV-based RCS measurement loader.

Expected CSV format (header required, column order flexible):

    theta_rad,phi_rad,freq_hz,rcs_real,rcs_imag

Optional column:
    snr_db   — per-point SNR estimate (used to set SCExtractorConfig.snr_db)

Example:
    theta_rad,phi_rad,freq_hz,rcs_real,rcs_imag
    0.314,0.0,8.0e9,0.0123,-0.0045
    0.628,0.0,9.0e9,-0.0082,0.0201
    ...

Usage::

    from dhff.io import RCSMeasurementLoader
    pts, vals = RCSMeasurementLoader("meas.csv").load()

    # Or use the convenience method:
    rcs = RCSMeasurementLoader("meas.csv").to_complex_rcs()
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from dhff.core.types import ComplexRCS, ObservationPoint

_REQUIRED_COLUMNS = {"theta_rad", "phi_rad", "freq_hz", "rcs_real", "rcs_imag"}


class RCSMeasurementLoader:
    """Load complex RCS measurements from a CSV file.

    Parameters
    ----------
    path:
        Path to the CSV file.
    freq_range_hz:
        If provided, rows with ``freq_hz`` outside this range are silently
        filtered out.
    """

    def __init__(
        self,
        path: str | Path,
        freq_range_hz: tuple[float, float] | None = None,
    ) -> None:
        self.path = Path(path)
        self.freq_range_hz = freq_range_hz
        self._obs: list[ObservationPoint] | None = None
        self._vals: np.ndarray | None = None
        self._snr_db: float | None = None

    # ------------------------------------------------------------------

    def load(self) -> tuple[list[ObservationPoint], np.ndarray]:
        """Parse the CSV and return ``(observation_points, complex_values)``.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If required column headers are missing or a data row is malformed.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"RCS CSV not found: {self.path}")

        with open(self.path, newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file appears empty: {self.path}")

            headers = {h.strip().lower() for h in reader.fieldnames}
            missing = _REQUIRED_COLUMNS - headers
            if missing:
                raise ValueError(
                    f"CSV is missing required columns: {sorted(missing)}. "
                    f"Found: {sorted(headers)}"
                )

            obs_list: list[ObservationPoint] = []
            val_list: list[complex] = []
            snr_list: list[float] = []
            has_snr = "snr_db" in headers

            for line_no, row in enumerate(reader, start=2):
                try:
                    theta = float(row["theta_rad"])
                    phi = float(row["phi_rad"])
                    freq = float(row["freq_hz"])
                    re = float(row["rcs_real"])
                    im = float(row["rcs_imag"])
                except (KeyError, ValueError) as exc:
                    raise ValueError(
                        f"Malformed row at line {line_no} in {self.path}: {exc}"
                    ) from exc

                if not (0.0 < theta < math.pi):
                    raise ValueError(
                        f"theta_rad={theta} out of range (0, π) at line {line_no}"
                    )
                if not (-2 * math.pi <= phi <= 2 * math.pi):
                    raise ValueError(
                        f"phi_rad={phi} out of range [-2π, 2π] at line {line_no}"
                    )
                if freq <= 0.0:
                    raise ValueError(
                        f"freq_hz={freq} must be positive at line {line_no}"
                    )

                if self.freq_range_hz is not None:
                    f_min, f_max = self.freq_range_hz
                    if not (f_min <= freq <= f_max):
                        continue

                obs_list.append(ObservationPoint(theta=theta, phi=phi, freq_hz=freq))
                val_list.append(complex(re, im))
                if has_snr:
                    try:
                        snr_list.append(float(row["snr_db"]))
                    except (KeyError, ValueError):
                        snr_list.append(float("nan"))

        if not obs_list:
            raise ValueError(
                f"No valid rows found in {self.path} "
                f"(freq_range_hz filter={self.freq_range_hz})"
            )

        self._obs = obs_list
        self._vals = np.array(val_list, dtype=np.complex128)
        if has_snr and snr_list:
            finite_snr = [s for s in snr_list if math.isfinite(s)]
            self._snr_db = float(np.median(finite_snr)) if finite_snr else None

        return self._obs, self._vals

    def to_complex_rcs(self) -> ComplexRCS:
        """Return a :class:`~dhff.core.types.ComplexRCS` object."""
        if self._obs is None:
            self.load()
        return ComplexRCS(observation_points=self._obs, values=self._vals)

    @property
    def median_snr_db(self) -> float | None:
        """Median SNR across all rows, or ``None`` if not provided in CSV."""
        if self._obs is None:
            self.load()
        return self._snr_db

    # ------------------------------------------------------------------
    # Helper: write a ComplexRCS to CSV (round-trip convenience)
    # ------------------------------------------------------------------

    @staticmethod
    def write_csv(
        path: str | Path,
        obs_points: list[ObservationPoint],
        values: np.ndarray,
        snr_db: float | None = None,
    ) -> None:
        """Write observation points + complex values to CSV.

        Parameters
        ----------
        path:
            Output file path.
        obs_points:
            List of :class:`ObservationPoint`.
        values:
            Complex array of RCS values, same length as ``obs_points``.
        snr_db:
            If provided, written as a constant ``snr_db`` column.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.asarray(values, dtype=np.complex128)
        with open(path, "w", newline="") as fh:
            fieldnames = ["theta_rad", "phi_rad", "freq_hz", "rcs_real", "rcs_imag"]
            if snr_db is not None:
                fieldnames.append("snr_db")
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for pt, v in zip(obs_points, values):
                row = {
                    "theta_rad": pt.theta,
                    "phi_rad": pt.phi,
                    "freq_hz": pt.freq_hz,
                    "rcs_real": v.real,
                    "rcs_imag": v.imag,
                }
                if snr_db is not None:
                    row["snr_db"] = snr_db
                writer.writerow(row)


class SimulationCSVLoader(RCSMeasurementLoader):
    """Same CSV format as :class:`RCSMeasurementLoader` but tagged as
    simulator output for documentation purposes."""
    pass
