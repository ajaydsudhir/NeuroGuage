"""Muse Acquisition Layer (Live/Replay/Synthetic).

This module bridges the Muse S Athena headband to the existing 5 inference
pipeline.  It provides three data sources that all share the same
:class:`MuseDataSource` protocol:

* :class:`LSLMuseSource` — subscribes to LSL outlets produced by
  `OpenMuse stream` and emits contract-shaped DataFrames (EEG, optical, IMU)
  into ring buffers consumed by the live app.
* :class:`SyntheticMuseSource` — generates realistic fake data for offline
  smoke tests, CI, or classroom demos without hardware.
* :class:`RawRecordingReplaySource` — replays a previously-recorded raw BLE
  packet log (``OpenMuse record`` output) through
  :func:`OpenMuse.decode_rawdata`, preserving the exact device stream layout.

Design notes
------------
* ``OpenMuse`` emits three LSL streams we care about:
  ``Muse-EEG (<addr>)`` (8 ch @ 256 Hz, EEG + AUX),
  ``Muse-OPTICS (<addr>)`` (16 ch @ 64 Hz for preset ``p1041``) and
  ``Muse-ACCGYRO (<addr>)`` (6 ch @ 52 Hz).  The :class:`MuseChannelMapping`
  below translates the OpenMuse channel labels into the column names required
  by ``configs/data_contract.json``.
* LSL streams are pulled with ``mne_lsl`` which is the current maintained
  Python binding (``pylsl`` is obsolete for our needs).
* The subprocess helper :class:`OpenMuseStreamProcess` wraps
  ``OpenMuse stream --address <mac> --preset p1041`` so the BLE bridge runs in
  its own process and our orchestrator only deals with LSL.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# Channel mapping — OpenMuse labels -> NeuroGauge data-contract column names.

# EEG labels per OpenMuse.decode.select_eeg_channels:
#   EEG8 = ['EEG_TP9', 'EEG_AF7', 'EEG_AF8', 'EEG_TP10',
#           'AUX_1', 'AUX_2', 'AUX_3', 'AUX_4']
# Contract (configs/data_contract.json -> eeg.required_columns):
#   eeg_tp9, eeg_af7, eeg_af8, eeg_tp10, eeg_aux
EEG_CHANNEL_MAP: dict[str, str] = {
    "EEG_TP9": "eeg_tp9",
    "EEG_AF7": "eeg_af7",
    "EEG_AF8": "eeg_af8",
    "EEG_TP10": "eeg_tp10",
    # AUX_1 is the bipolar amplified aux line typically used for peripheral EEG.
    "AUX_1": "eeg_aux",
}

# Optical mapping — OpenMuse preset p1041 delivers 16 channels at 64 Hz:
#   OPTICS_{LO|RO|LI|RI}_{NIR|IR|RED|AMB}
# Our contract only requires the outer (LO/RO) optodes for all 4 wavelengths.
# We also carry the inner (LI/RI) channels in the contract-plus DataFrame so
# downstream modules can use the richer Athena signal if desired.
OPTICAL_CHANNEL_MAP: dict[str, str] = {
    "OPTICS_LO_NIR": "optics_lo_nir",
    "OPTICS_RO_NIR": "optics_ro_nir",
    "OPTICS_LO_IR":  "optics_lo_ir",
    "OPTICS_RO_IR":  "optics_ro_ir",
    "OPTICS_LO_RED": "optics_lo_red",
    "OPTICS_RO_RED": "optics_ro_red",
    "OPTICS_LO_AMB": "optics_lo_amb",
    "OPTICS_RO_AMB": "optics_ro_amb",
    # Inner optodes — exposed as optics_* but not in the strict contract list.
    "OPTICS_LI_NIR": "optics_li_nir",
    "OPTICS_RI_NIR": "optics_ri_nir",
    "OPTICS_LI_IR":  "optics_li_ir",
    "OPTICS_RI_IR":  "optics_ri_ir",
    "OPTICS_LI_RED": "optics_li_red",
    "OPTICS_RI_RED": "optics_ri_red",
    "OPTICS_LI_AMB": "optics_li_amb",
    "OPTICS_RI_AMB": "optics_ri_amb",
}

IMU_CHANNEL_MAP: dict[str, str] = {
    "ACC_X": "acc_x",
    "ACC_Y": "acc_y",
    "ACC_Z": "acc_z",
    "GYRO_X": "gyro_x",
    "GYRO_Y": "gyro_y",
    "GYRO_Z": "gyro_z",
}

# Contract sampling frequencies (used by synthetic/replay sources).
EEG_SFREQ_HZ = 256.0
OPTICAL_SFREQ_HZ = 64.0
IMU_SFREQ_HZ = 52.0

# Columns returned by each read_frames() call (always populated in order).
EEG_OUTPUT_COLUMNS = [
    "time_s", "session_id", "timestamp_utc",
    "eeg_tp9", "eeg_af7", "eeg_af8", "eeg_tp10", "eeg_aux",
]
OPTICAL_OUTPUT_COLUMNS = [
    "time_s", "session_id", "timestamp_utc",
    "optics_lo_nir", "optics_ro_nir", "optics_lo_ir", "optics_ro_ir",
    "optics_lo_red", "optics_ro_red", "optics_lo_amb", "optics_ro_amb",
]
IMU_OUTPUT_COLUMNS = [
    "time_s", "session_id", "timestamp_utc",
    "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
]


# Data-source protocol

@dataclass
class MuseFrames:
    """Three synchronized DataFrames returned by a single read."""

    eeg: pd.DataFrame
    optical: pd.DataFrame
    imu: pd.DataFrame

    @property
    def empty(self) -> bool:
        return self.eeg.empty and self.optical.empty and self.imu.empty


class MuseDataSource(Protocol):
    """Interface every Muse data source must implement."""

    source_name: str
    session_id: str

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read_frames(self) -> MuseFrames: ...
    def health(self) -> dict[str, Any]: ...


# OpenMuse subprocess helper

@dataclass
class OpenMuseStreamProcess:
    """Spawn the ``OpenMuse stream`` CLI as a child process.

    Running OpenMuse's BLE + LSL bridge out-of-process is cleaner than calling
    ``OpenMuse.stream`` directly because that function uses ``asyncio.run``
    and therefore cannot coexist with our matplotlib event loop.
    """

    address: str
    preset: str = "p1041"
    duration: Optional[float] = None
    record_path: Optional[str] = None
    python_executable: str = field(default_factory=lambda: sys.executable)
    verbose: bool = True
    _process: Optional[subprocess.Popen] = None
    _stopped: bool = False

    def _resolve_command(self) -> list[str]:
        """Return the best-available invocation for ``OpenMuse stream``.

        Priority order:
        1. ``OpenMuse`` console script on PATH (created by ``pip install``).
        2. ``<env>/Scripts/OpenMuse.exe`` next to ``python.exe`` on Windows.
        3. ``<env>/bin/OpenMuse`` on POSIX.
        4. ``python -m OpenMuse`` — only works if a future OpenMuse release
           adds ``__main__.py``; currently raises but kept as a last resort
           for forward compatibility.
        """
        on_path = shutil.which("OpenMuse")
        if on_path:
            return [on_path]
        py_dir = Path(self.python_executable).resolve().parent
        for candidate in (
            py_dir / "Scripts" / "OpenMuse.exe",   # Windows
            py_dir / "Scripts" / "OpenMuse",       # Windows w/o ext
            py_dir / "OpenMuse.exe",
            py_dir / "bin" / "OpenMuse",           # POSIX
            py_dir.parent / "Scripts" / "OpenMuse.exe",
            py_dir.parent / "bin" / "OpenMuse",
        ):
            if candidate.exists():
                return [str(candidate)]
        LOGGER.warning(
            "OpenMuse console script not found; falling back to "
            "`python -m OpenMuse` which requires OpenMuse>=<future>."
        )
        return [self.python_executable, "-m", "OpenMuse"]

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("OpenMuseStreamProcess already started")

        cmd: list[str] = [
            *self._resolve_command(),
            "stream",
            "--address", self.address,
            "--preset", self.preset,
        ]
        if self.duration is not None:
            cmd += ["--duration", f"{self.duration:.0f}"]
        if self.record_path is not None:
            record_path = Path(self.record_path)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            cmd += ["--record", str(record_path)]

        LOGGER.info("Spawning OpenMuse bridge: %s", " ".join(cmd))
        self._process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE if not self.verbose else None,
            stderr=subprocess.STDOUT if not self.verbose else None,
        )

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def returncode(self) -> Optional[int]:
        if self._process is None:
            return None
        return self._process.poll()

    def stop(self, timeout_s: float = 5.0) -> None:
        if self._process is None or self._stopped:
            return
        try:
            self._process.terminate()
            self._process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            LOGGER.warning("OpenMuse bridge did not terminate; killing.")
            self._process.kill()
        finally:
            self._stopped = True

    def __enter__(self) -> "OpenMuseStreamProcess":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def discover_muse_devices(timeout: float = 10.0, verbose: bool = True) -> list[dict[str, Any]]:
    """Scan for Muse devices over BLE (wraps ``OpenMuse.find_muse``).

    Returns a list of ``{"name": ..., "address": ...}`` dicts.  The list will
    be empty when no device is paired or BLE is unavailable.  This function
    never raises — callers should log the empty-list case and instruct the
    user to power the headband on.
    """
    try:
        import OpenMuse  # noqa: PLC0415 — optional runtime dep
    except Exception as exc:  # pragma: no cover — import error path
        LOGGER.warning("OpenMuse is not installed: %s", exc)
        return []

    try:
        return list(OpenMuse.find_muse(timeout=timeout, verbose=verbose))
    except Exception as exc:
        LOGGER.warning("Muse discovery failed: %s", exc)
        return []


# LSL-backed live source

@dataclass
class LSLMuseSource:
    """Subscribe to OpenMuse LSL outlets and emit contract-shaped DataFrames.

    The source can either *own* an :class:`OpenMuseStreamProcess` (it will
    spawn the BLE bridge) or assume a stream is already running (useful when
    the user ran ``OpenMuse stream`` manually in another terminal).
    """

    device_address: Optional[str] = None
    session_id: str = "live"
    preset: str = "p1041"
    manage_bridge: bool = True
    resolve_timeout_s: float = 60.0
    resolve_poll_interval_s: float = 2.0
    max_pull_per_read: int = 4096
    pull_timeout_s: float = 0.0
    record_raw_to: Optional[str] = None
    require_modalities: tuple[str, ...] = ("eeg", "optical", "imu")
    require_full_link: bool = True
    bridge_warmup_s: float = 5.0

    source_name: str = field(default="muse_lsl", init=False)
    _bridge: Optional[OpenMuseStreamProcess] = field(default=None, init=False)
    _inlets: dict[str, Any] = field(default_factory=dict, init=False)
    _inlet_channel_labels: dict[str, list[str]] = field(default_factory=dict, init=False)
    _reference_time: Optional[float] = field(default=None, init=False)
    _last_lsl_time: dict[str, float] = field(default_factory=dict, init=False)
    _last_wall_time: dict[str, float] = field(default_factory=dict, init=False)
    _samples_received: dict[str, int] = field(default_factory=dict, init=False)
    _started: bool = field(default=False, init=False)

    def start(self) -> None:
        if self._started:
            return

        if self.manage_bridge:
            if self.device_address is None:
                raise ValueError("device_address is required when manage_bridge=True")
            self._bridge = OpenMuseStreamProcess(
                address=self.device_address,
                preset=self.preset,
                record_path=self.record_raw_to,
            )
            self._bridge.start()
            # BLE connect + preset handshake on Muse S Athena typically takes
            # 8-15 s before the first packet of each sensor type arrives, and
            # OpenMuse only creates each LSL outlet on the first packet.
            LOGGER.info(
                "Waiting %.1fs for OpenMuse BLE bridge to warm up...",
                self.bridge_warmup_s,
            )
            time.sleep(self.bridge_warmup_s)
            if not self._bridge.is_alive():
                rc = self._bridge.returncode
                raise RuntimeError(
                    "OpenMuse bridge subprocess exited during warmup "
                    f"(returncode={rc}). Verify that the `OpenMuse` console "
                    "script is installed (try running `OpenMuse find` "
                    "manually) and that the headband is powered on."
                )

        self._resolve_streams()
        self._reference_time = time.time()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        for inlet in self._inlets.values():
            try:
                inlet.close_stream()
            except Exception:
                pass
        self._inlets.clear()
        if self._bridge is not None:
            self._bridge.stop()
            self._bridge = None
        self._started = False

    def _resolve_streams(self) -> None:
        from mne_lsl.lsl import (  # noqa: PLC0415 — optional runtime dep
            StreamInlet,
            resolve_streams,
        )

        # OpenMuse names: 'Muse-EEG (<addr>)' stype='EEG',
        # 'Muse-OPTICS (<addr>)' stype='PPG',
        # 'Muse-ACCGYRO (<addr>)' stype='ACCGYRO',
        # 'Muse-BATTERY (<addr>)' stype='Battery'.
        wanted_stypes = {"eeg": "EEG", "optical": "PPG", "imu": "ACCGYRO"}
        deadline = time.time() + self.resolve_timeout_s

        while True:
            try:
                infos = resolve_streams(timeout=self.resolve_poll_interval_s)
            except Exception as exc:
                LOGGER.warning("resolve_streams failed: %s", exc)
                infos = []

            for modality, stype in wanted_stypes.items():
                if modality in self._inlets:
                    continue
                candidates = [info for info in infos if (info.stype or "") == stype]
                if self.device_address:
                    matching = [
                        info for info in candidates
                        if self.device_address in (info.source_id or "")
                        or self.device_address in (info.name or "")
                    ]
                    if matching:
                        candidates = matching
                if not candidates:
                    continue

                info = candidates[0]
                inlet = StreamInlet(info, max_buffered=60, processing_flags=None)
                try:
                    inlet.open_stream(timeout=self.resolve_poll_interval_s)
                except Exception as exc:
                    LOGGER.warning(
                        "open_stream failed for modality=%s name=%s: %s",
                        modality, info.name, exc,
                    )
                    continue

                labels: list[str] = []
                try:
                    for ch in info.get_channel_names() or []:
                        if ch is not None:
                            labels.append(str(ch))
                except Exception:
                    pass
                if not labels:
                    if modality == "eeg":
                        labels = ["EEG_TP9", "EEG_AF7", "EEG_AF8", "EEG_TP10",
                                  "AUX_1", "AUX_2", "AUX_3", "AUX_4"][:info.n_channels]
                    elif modality == "optical":
                        labels = list(OPTICAL_CHANNEL_MAP.keys())[:info.n_channels]
                    elif modality == "imu":
                        labels = ["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"]

                LOGGER.info(
                    "Resolved LSL stream | modality=%s | name=%s | stype=%s | sfreq=%.1f | ch=%d",
                    modality, info.name, info.stype, info.sfreq, info.n_channels,
                )
                self._inlets[modality] = inlet
                self._inlet_channel_labels[modality] = labels
                self._samples_received[modality] = 0

            missing = [m for m in self.require_modalities if m not in self._inlets]
            if not missing:
                return
            if time.time() >= deadline:
                if self.require_full_link:
                    raise RuntimeError(
                        "Muse link not ready: timed out waiting for required LSL "
                        f"modalities {missing} after {self.resolve_timeout_s:.1f}s. "
                        "NeuroGauge will not start until the Muse S Athena is fully "
                        "linked and streaming EEG, optical, and IMU data."
                    )
                LOGGER.warning(
                    "LSL resolve timed out after %.1fs; missing modalities=%s. "
                    "Pipeline will run with partial data.",
                    self.resolve_timeout_s, missing,
                )
                return

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def _pull_modality(self, modality: str) -> tuple[np.ndarray, np.ndarray]:
        inlet = self._inlets.get(modality)
        if inlet is None:
            return np.empty((0, 0)), np.empty((0,))
        try:
            samples, timestamps = inlet.pull_chunk(
                timeout=self.pull_timeout_s,
                max_samples=self.max_pull_per_read,
            )
        except Exception as exc:
            LOGGER.warning("Pull error on %s: %s", modality, exc)
            return np.empty((0, 0)), np.empty((0,))

        samples = np.asarray(samples, dtype=np.float64) if len(samples) else np.empty((0, 0))
        timestamps = np.asarray(timestamps, dtype=np.float64) if len(timestamps) else np.empty((0,))
        if len(timestamps):
            self._last_lsl_time[modality] = float(timestamps[-1])
            self._last_wall_time[modality] = time.time()
            self._samples_received[modality] = self._samples_received.get(modality, 0) + len(timestamps)
        return samples, timestamps

    def _to_frame(
        self,
        modality: str,
        samples: np.ndarray,
        timestamps: np.ndarray,
        mapping: dict[str, str],
        output_columns: list[str],
    ) -> pd.DataFrame:
        if samples.size == 0 or timestamps.size == 0:
            return pd.DataFrame(columns=output_columns)

        labels = self._inlet_channel_labels.get(modality, [])
        columns: list[str] = []
        keep_idx: list[int] = []
        for idx, label in enumerate(labels):
            contract_col = mapping.get(label)
            if contract_col is None:
                continue
            columns.append(contract_col)
            keep_idx.append(idx)

        if not keep_idx:
            return pd.DataFrame(columns=output_columns)

        base = pd.DataFrame(samples[:, keep_idx], columns=columns)
        base["time_s"] = timestamps - (self._reference_time or timestamps[0])
        base["session_id"] = self.session_id
        base["timestamp_utc"] = pd.to_datetime(timestamps, unit="s", utc=True, errors="coerce")

        # Ensure all contract columns exist (fill missing with NaN).
        for col in output_columns:
            if col not in base.columns:
                base[col] = np.nan
        return base[output_columns]

    def read_frames(self) -> MuseFrames:
        if not self._started:
            return MuseFrames(
                eeg=pd.DataFrame(columns=EEG_OUTPUT_COLUMNS),
                optical=pd.DataFrame(columns=OPTICAL_OUTPUT_COLUMNS),
                imu=pd.DataFrame(columns=IMU_OUTPUT_COLUMNS),
            )

        eeg_samples, eeg_ts = self._pull_modality("eeg")
        opt_samples, opt_ts = self._pull_modality("optical")
        imu_samples, imu_ts = self._pull_modality("imu")

        return MuseFrames(
            eeg=self._to_frame("eeg", eeg_samples, eeg_ts, EEG_CHANNEL_MAP, EEG_OUTPUT_COLUMNS),
            optical=self._to_frame("optical", opt_samples, opt_ts, OPTICAL_CHANNEL_MAP, OPTICAL_OUTPUT_COLUMNS),
            imu=self._to_frame("imu", imu_samples, imu_ts, IMU_CHANNEL_MAP, IMU_OUTPUT_COLUMNS),
        )

    def health(self) -> dict[str, Any]:
        now = time.time()
        info: dict[str, Any] = {
            "source": self.source_name,
            "started": self._started,
            "bridge_alive": bool(self._bridge and self._bridge.is_alive()),
            "samples": dict(self._samples_received),
            "last_lsl_time": dict(self._last_lsl_time),
            "last_wall_time": dict(self._last_wall_time),
            "wall_time": now,
            "reference_time": self._reference_time,
        }
        for modality in ("eeg", "optical", "imu"):
            wall_last = self._last_wall_time.get(modality)
            info[f"{modality}_healthy"] = (
                wall_last is not None and (now - wall_last) < 5.0
            )
        return info


# Synthetic source — useful for CI, tutorials, and classroom fallbacks.

@dataclass
class SyntheticMuseSource:
    """Generate plausible Muse-like signals without hardware.

    The signals are *not* biologically faithful but preserve sampling rates,
    channel counts, rough amplitudes, and a mild fatigue trend so the full
    NeuroGauge pipeline can be exercised without a device.
    """

    session_id: str = "synthetic"
    fatigue_trend: float = 0.5           # 0.0 = alert, 1.0 = drowsy
    rng_seed: int = 42
    real_time_factor: float = 1.0        # >1 = faster than wall clock

    source_name: str = field(default="synthetic", init=False)
    _rng: np.random.Generator = field(default=None, init=False)  # type: ignore[assignment]
    _t0: float = field(default=0.0, init=False)
    _last_read: float = field(default=0.0, init=False)
    _started: bool = field(default=False, init=False)
    _last_counts: dict[str, int] = field(default_factory=dict, init=False)

    def start(self) -> None:
        self._rng = np.random.default_rng(self.rng_seed)
        self._t0 = time.time()
        self._last_read = self._t0
        self._started = True
        self._last_counts = {"eeg": 0, "optical": 0, "imu": 0}

    def stop(self) -> None:
        self._started = False

    def _span_s(self) -> tuple[float, float]:
        now = time.time()
        span_start = (self._last_read - self._t0) * self.real_time_factor
        span_end = (now - self._t0) * self.real_time_factor
        self._last_read = now
        return span_start, span_end

    def _make_modality(
        self,
        span_s: tuple[float, float],
        sfreq: float,
        channels: list[str],
        generator: Any,
        output_columns: list[str],
    ) -> pd.DataFrame:
        start_s, end_s = span_s
        duration = max(0.0, end_s - start_s)
        n_samples = int(duration * sfreq)
        if n_samples <= 0:
            return pd.DataFrame(columns=output_columns)

        t = start_s + np.arange(n_samples) / sfreq
        data = generator(t, channels)
        frame = pd.DataFrame(data, columns=channels)
        frame["time_s"] = t
        frame["session_id"] = self.session_id
        frame["timestamp_utc"] = pd.to_datetime(
            self._t0 + t / max(self.real_time_factor, 1e-9),
            unit="s", utc=True, errors="coerce",
        )
        for col in output_columns:
            if col not in frame.columns:
                frame[col] = np.nan
        return frame[output_columns]

    def _gen_eeg(self, t: np.ndarray, channels: list[str]) -> dict[str, np.ndarray]:
        # Alpha (10 Hz) dominates at rest; theta (6 Hz) grows with fatigue;
        # beta (20 Hz) attenuates with fatigue.
        fatigue = float(np.clip(self.fatigue_trend, 0.0, 1.0))
        alpha_amp = 18.0 * (1.0 - 0.3 * fatigue)
        theta_amp = 8.0 + 22.0 * fatigue
        beta_amp  = 10.0 * (1.0 - 0.4 * fatigue)
        out: dict[str, np.ndarray] = {}
        for ch in channels:
            phase = self._rng.uniform(0, 2 * np.pi)
            noise = self._rng.standard_normal(len(t)) * 4.0
            drift = self._rng.standard_normal(1) * 0.5
            signal = (
                alpha_amp * np.sin(2 * np.pi * 10.0 * t + phase)
                + theta_amp * np.sin(2 * np.pi * 6.2 * t + phase * 1.3)
                + beta_amp * np.sin(2 * np.pi * 20.0 * t + phase * 0.7)
                + noise + drift
            )
            out[ch] = signal
        return out

    def _gen_optical(self, t: np.ndarray, channels: list[str]) -> dict[str, np.ndarray]:
        # Slow hemodynamic drift (~0.1 Hz) + small respiratory component.
        fatigue = float(np.clip(self.fatigue_trend, 0.0, 1.0))
        out: dict[str, np.ndarray] = {}
        for ch in channels:
            base = 1.0 + 0.02 * np.sin(2 * np.pi * 0.1 * t + self._rng.uniform())
            resp = 0.005 * np.sin(2 * np.pi * 0.25 * t)
            fatigue_drift = fatigue * 0.01 * t / max(1.0, float(t[-1] - t[0] + 1e-9))
            noise = self._rng.standard_normal(len(t)) * 0.002
            out[ch] = base + resp + fatigue_drift + noise
        return out

    def _gen_imu(self, t: np.ndarray, channels: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for ch in channels:
            if ch.startswith("acc_"):
                bias = {"acc_x": 0.0, "acc_y": 0.0, "acc_z": 0.98}.get(ch, 0.0)
                out[ch] = bias + self._rng.standard_normal(len(t)) * 0.01
            else:
                out[ch] = self._rng.standard_normal(len(t)) * 0.5
        return out

    def read_frames(self) -> MuseFrames:
        if not self._started:
            return MuseFrames(
                eeg=pd.DataFrame(columns=EEG_OUTPUT_COLUMNS),
                optical=pd.DataFrame(columns=OPTICAL_OUTPUT_COLUMNS),
                imu=pd.DataFrame(columns=IMU_OUTPUT_COLUMNS),
            )
        span = self._span_s()
        eeg_channels = ["eeg_tp9", "eeg_af7", "eeg_af8", "eeg_tp10", "eeg_aux"]
        opt_channels = [c for c in OPTICAL_OUTPUT_COLUMNS
                        if c not in {"time_s", "session_id", "timestamp_utc"}]
        imu_channels = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

        eeg = self._make_modality(span, EEG_SFREQ_HZ, eeg_channels, self._gen_eeg, EEG_OUTPUT_COLUMNS)
        opt = self._make_modality(span, OPTICAL_SFREQ_HZ, opt_channels, self._gen_optical, OPTICAL_OUTPUT_COLUMNS)
        imu = self._make_modality(span, IMU_SFREQ_HZ, imu_channels, self._gen_imu, IMU_OUTPUT_COLUMNS)

        self._last_counts["eeg"] += len(eeg)
        self._last_counts["optical"] += len(opt)
        self._last_counts["imu"] += len(imu)

        return MuseFrames(eeg=eeg, optical=opt, imu=imu)

    def health(self) -> dict[str, Any]:
        return {
            "source": self.source_name,
            "started": self._started,
            "samples": dict(self._last_counts),
            "fatigue_trend": self.fatigue_trend,
            "real_time_factor": self.real_time_factor,
        }


# Recorded-raw replay source (OpenMuse record --> decode_rawdata)

@dataclass
class RawRecordingReplaySource:
    """Replay a previously recorded raw BLE packet file through the pipeline.

    The file must be a line-separated UTF-8 text file produced by
    ``OpenMuse record`` (or ``OpenMuse stream --record ...``).  We decode the
    full session into the three contract modalities up-front, then hand out
    slices of that session whenever the orchestrator asks for new data.
    """

    raw_file: str | Path
    session_id: str = "replay"
    real_time_factor: float = 1.0

    source_name: str = field(default="raw_replay", init=False)
    _frames: Optional[MuseFrames] = field(default=None, init=False)
    _index: dict[str, int] = field(default_factory=dict, init=False)
    _t0: float = field(default=0.0, init=False)
    _started: bool = field(default=False, init=False)
    _max_time_s: float = field(default=0.0, init=False)

    def start(self) -> None:
        self._frames = self._decode_session()
        self._index = {"eeg": 0, "optical": 0, "imu": 0}
        self._t0 = time.time()
        self._started = True

    def stop(self) -> None:
        self._started = False

    def _decode_session(self) -> MuseFrames:
        path = Path(self.raw_file)
        if not path.exists():
            raise FileNotFoundError(f"Raw Muse recording not found: {path}")

        try:
            import OpenMuse  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "OpenMuse is required for RawRecordingReplaySource. "
                "Run `pip install -r requirements.txt` first."
            ) from exc

        with path.open("r", encoding="utf-8") as handle:
            messages = handle.readlines()
        decoded = OpenMuse.decode_rawdata(messages)

        eeg = self._decode_eeg_frame(decoded)
        optical = self._decode_optical_frame(decoded)
        imu = self._decode_imu_frame(decoded)

        for df in (eeg, optical, imu):
            if not df.empty:
                self._max_time_s = max(self._max_time_s, float(df["time_s"].max()))
        return MuseFrames(eeg=eeg, optical=optical, imu=imu)

    def _prep_frame(
        self,
        frame: pd.DataFrame,
        mapping: dict[str, str],
        output_columns: list[str],
    ) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=output_columns)

        # ``decode_rawdata`` yields a "time" column in seconds.  Our pipeline
        # uses ``time_s`` throughout.
        renamed = frame.rename(columns=mapping)
        if "time" in renamed.columns and "time_s" not in renamed.columns:
            renamed["time_s"] = renamed["time"]
        # Drop any raw OpenMuse columns that we don't carry forward.
        for col in output_columns:
            if col not in renamed.columns:
                renamed[col] = np.nan
        renamed["session_id"] = self.session_id
        if "timestamp_utc" not in renamed.columns or renamed["timestamp_utc"].isna().all():
            renamed["timestamp_utc"] = pd.to_datetime(
                renamed["time_s"], unit="s", utc=True, errors="coerce",
            )
        return renamed[output_columns].copy()

    def _decode_eeg_frame(self, decoded: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frame = decoded.get("EEG")
        if frame is None or frame.empty:
            return pd.DataFrame(columns=EEG_OUTPUT_COLUMNS)
        return self._prep_frame(frame, EEG_CHANNEL_MAP, EEG_OUTPUT_COLUMNS)

    def _decode_optical_frame(self, decoded: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frame = decoded.get("OPTICS")
        if frame is None or frame.empty:
            return pd.DataFrame(columns=OPTICAL_OUTPUT_COLUMNS)
        return self._prep_frame(frame, OPTICAL_CHANNEL_MAP, OPTICAL_OUTPUT_COLUMNS)

    def _decode_imu_frame(self, decoded: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frame = decoded.get("ACCGYRO")
        if frame is None or frame.empty:
            return pd.DataFrame(columns=IMU_OUTPUT_COLUMNS)
        return self._prep_frame(frame, IMU_CHANNEL_MAP, IMU_OUTPUT_COLUMNS)

    def _slice_until(self, modality: str, frame: pd.DataFrame, elapsed_s: float) -> pd.DataFrame:
        start = self._index.get(modality, 0)
        if frame is None or frame.empty or start >= len(frame):
            return frame.iloc[0:0] if frame is not None else pd.DataFrame()
        end = start
        times = frame["time_s"].to_numpy()
        while end < len(frame) and times[end] <= elapsed_s:
            end += 1
        self._index[modality] = end
        return frame.iloc[start:end].copy()

    def read_frames(self) -> MuseFrames:
        if not self._started or self._frames is None:
            return MuseFrames(
                eeg=pd.DataFrame(columns=EEG_OUTPUT_COLUMNS),
                optical=pd.DataFrame(columns=OPTICAL_OUTPUT_COLUMNS),
                imu=pd.DataFrame(columns=IMU_OUTPUT_COLUMNS),
            )

        wall_elapsed = time.time() - self._t0
        session_elapsed = wall_elapsed * max(self.real_time_factor, 1e-9)

        return MuseFrames(
            eeg=self._slice_until("eeg", self._frames.eeg, session_elapsed),
            optical=self._slice_until("optical", self._frames.optical, session_elapsed),
            imu=self._slice_until("imu", self._frames.imu, session_elapsed),
        )

    @property
    def finished(self) -> bool:
        return self._started and all(
            self._index.get(m, 0) >= (len(getattr(self._frames, m)) if self._frames else 0)
            for m in ("eeg", "optical", "imu")
        )

    def health(self) -> dict[str, Any]:
        return {
            "source": self.source_name,
            "started": self._started,
            "samples_served": dict(self._index),
            "session_length_s": self._max_time_s,
            "raw_file": str(self.raw_file),
        }


# Factory

def build_data_source(
    *,
    source_kind: str,
    session_id: str = "live",
    device_address: Optional[str] = None,
    preset: str = "p1041",
    raw_file: Optional[str] = None,
    manage_bridge: bool = True,
    real_time_factor: float = 1.0,
    fatigue_trend: float = 0.5,
    record_raw_to: Optional[str] = None,
    resolve_timeout_s: float = 60.0,
) -> MuseDataSource:
    """Return a configured :class:`MuseDataSource` for the chosen backend."""
    kind = source_kind.lower()
    if kind in ("muse", "lsl", "openmuse"):
        if manage_bridge and device_address is None:
            raise ValueError(
                "device_address is required for source_kind='muse' when "
                "manage_bridge=True. Pass manage_bridge=False to attach to "
                "an externally-managed `OpenMuse stream` session."
            )
        return LSLMuseSource(
            device_address=device_address,
            session_id=session_id,
            preset=preset,
            manage_bridge=manage_bridge,
            record_raw_to=record_raw_to,
            resolve_timeout_s=resolve_timeout_s,
            require_full_link=True,
        )
    if kind in ("synthetic", "fake", "simulated"):
        return SyntheticMuseSource(
            session_id=session_id,
            fatigue_trend=fatigue_trend,
            real_time_factor=real_time_factor,
        )
    if kind in ("replay", "raw", "rawfile"):
        if not raw_file:
            raise ValueError("raw_file is required for source_kind='replay'")
        return RawRecordingReplaySource(
            raw_file=raw_file,
            session_id=session_id,
            real_time_factor=real_time_factor,
        )
    raise ValueError(f"Unknown source_kind: {source_kind}")


# Session recorder — decode a raw file into contract-shaped CSVs.

def decode_raw_session_to_contract(
    raw_file: str | Path,
    session_id: str,
    output_dir: str | Path,
    task_type: str = "sustained_attention",
    subjective_fatigue_score: float = 5.0,
) -> dict[str, Path]:
    """Convert an ``OpenMuse record`` .txt file into session-contract CSVs.

    Writes ``<output_dir>/{eeg,optical,imu}.csv`` plus a ``metadata.json``
    that conforms to ``configs/data_contract.json``.  Returns a mapping of
    modality -> path for downstream consumers (schema validator, label
    generator, feature builder).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = RawRecordingReplaySource(raw_file=raw_file, session_id=session_id)
    source.start()
    assert source._frames is not None  # set by start()
    frames = source._frames

    contract_paths: dict[str, Path] = {}
    for modality, df, columns in [
        ("eeg", frames.eeg, EEG_OUTPUT_COLUMNS),
        ("optical", frames.optical, OPTICAL_OUTPUT_COLUMNS),
        ("imu", frames.imu, IMU_OUTPUT_COLUMNS),
    ]:
        path = out_dir / f"{modality}.csv"
        export = df.copy()
        if "timestamp_utc" in export.columns:
            export["timestamp_utc"] = (
                export["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                if hasattr(export["timestamp_utc"], "dt")
                else export["timestamp_utc"].astype(str)
            )
        # Drop helper columns that aren't in the contract.
        export = export[[c for c in columns if c in export.columns]]
        export.to_csv(path, index=False)
        contract_paths[modality] = path

    now_utc = pd.Timestamp.now(tz="UTC")
    start_utc = now_utc.isoformat()
    end_utc = (now_utc + pd.Timedelta(seconds=source._max_time_s)).isoformat()
    metadata = {
        "session_id": session_id,
        "task_type": task_type,
        "start_timestamp_utc": start_utc,
        "end_timestamp_utc": end_utc,
        "subjective_fatigue_score": subjective_fatigue_score,
        "reaction_time_ms": None,
        "accuracy": None,
        "source_recording": str(Path(raw_file).resolve()),
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    contract_paths["metadata"] = metadata_path

    source.stop()
    LOGGER.info("Decoded raw session to %s", out_dir)
    return contract_paths


__all__ = [
    "EEG_CHANNEL_MAP",
    "EEG_OUTPUT_COLUMNS",
    "EEG_SFREQ_HZ",
    "IMU_CHANNEL_MAP",
    "IMU_OUTPUT_COLUMNS",
    "IMU_SFREQ_HZ",
    "LSLMuseSource",
    "MuseDataSource",
    "MuseFrames",
    "OPTICAL_CHANNEL_MAP",
    "OPTICAL_OUTPUT_COLUMNS",
    "OPTICAL_SFREQ_HZ",
    "OpenMuseStreamProcess",
    "RawRecordingReplaySource",
    "SyntheticMuseSource",
    "build_data_source",
    "decode_raw_session_to_contract",
    "discover_muse_devices",
]
