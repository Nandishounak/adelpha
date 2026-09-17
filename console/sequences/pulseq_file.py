"""Play a .seq file staged with ``scripts/load_seq.py``.

The script checks a file and copies it into ``<MRI4ALL_BASE>/seq_library/``.
This sequence plays whichever library file ``param_seq_file`` names; the
default ``latest`` is the most recently loaded one, so the usual flow is just
"add Run .seq file, then scan".
"""

import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import common.logger as logger
import external.seq.adjustments_acq.config as cfg
from common import runtime
from common.types import ResultItem
from external.seq.adjustments_acq.scripts import run_pulseq
from sequences import PulseqSequence, param  # type: ignore

log = logger.get_logger()

LIBRARY_DIRNAME = "seq_library"
LATEST_POINTER = "LATEST"

_PANELS = (
    ("RF", "|tx| (DAC)", ("tx0",)),
    ("Receive", "gate", ("rx0_en",)),
    ("Gradients", "DAC", ("grad_vx", "grad_vy", "grad_vz", "grad_vz2")),
)


def instructions_figure(channels: Dict[str, Tuple[np.ndarray, np.ndarray]], title: str = ""):
    """Plot MaRCoS instructions as they are played: each value held until the next.

    ``channels`` maps channel name -> (times in microseconds, values). Steps are
    expanded into explicit points so viewers that join points with straight
    lines still show held values rather than invented ramps.
    """
    fig, axes = plt.subplots(len(_PANELS), 1, sharex=True, figsize=(10, 7), constrained_layout=True)
    for ax, (panel, ylabel, names) in zip(axes, _PANELS):
        for name in names:
            if name not in channels:
                continue
            t_us, values = channels[name]
            values = np.abs(values) if name == "tx0" else np.real(values)
            if len(t_us) < 2 or (name.startswith("grad") and not np.any(values)):
                continue
            t_ms = np.asarray(t_us, dtype=float) / 1e3
            ax.plot(np.repeat(t_ms, 2)[1:], np.repeat(values, 2)[:-1], lw=0.8, label=name)
        ax.set_title(f"{title} — {panel}" if title else panel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if ax.get_lines():
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("time (ms)")
    return fig


def _channels_from_axes(fig) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Recover the channels run_pulseq plotted (it labels each line by channel)."""
    channels = {}
    for ax in fig.axes:
        for line in ax.get_lines():
            label = line.get_label()
            if not label.startswith("_"):
                x, y = line.get_data()
                channels[label] = (np.asarray(x), np.asarray(y))
    return channels


def library_dir() -> Path:
    return Path(runtime.get_base_path()) / LIBRARY_DIRNAME


def resolve_seq_file(name: str) -> Optional[Path]:
    """Map a ``param_seq_file`` value to a file in the library, or None."""
    library = library_dir()
    name = (name or "latest").strip()
    if name.lower() == "latest":
        pointer = library / LATEST_POINTER
        if not pointer.is_file():
            return None
        name = pointer.read_text(encoding="utf-8").strip()
    if not name.endswith(".seq"):
        name += ".seq"
    candidate = library / Path(name).name  # never escape the library
    return candidate if candidate.is_file() else None


class SequencePulseqFile(PulseqSequence, registry_key=Path(__file__).stem):
    param_seq_file = param(
        "latest",
        title="Sequence file",
        description="File name in seq_library, or 'latest' for the last one loaded",
    )

    @classmethod
    def get_readable_name(cls) -> str:
        return "Run .seq file"

    @classmethod
    def get_description(cls) -> str:
        return "Plays a .seq file loaded with scripts/load_seq.py"

    def validate_parameters(self, scan_task) -> bool:
        if resolve_seq_file(self.param_seq_file) is None:
            self.problem_list.append(
                f"No '{self.param_seq_file}' in {library_dir()}. "
                "Load one with scripts/load_seq.py."
            )
        return self.is_valid()

    def calculate_sequence(self, scan_task) -> bool:
        source = resolve_seq_file(self.param_seq_file)
        if source is None:
            log.error("Sequence file '%s' not found in %s", self.param_seq_file, library_dir())
            return False

        scan_task.processing.recon_mode = "bypass"
        self.seq_file_path = self.get_working_folder() + "/seq/acq0.seq"
        # Copy rather than reference, so the scan folder keeps what was actually played.
        shutil.copyfile(source, self.seq_file_path)
        log.info("Using %s for %s", source.name, self.get_name())
        self.calculated = True
        return True

    def _attach_sequence_plot(self, scan_task) -> None:
        """Show what MaRCoS was sent, using run_pulseq's own interpretation."""
        try:
            channels = _channels_from_axes(plt.gcf())
            plt.close("all")
            if not channels:
                return
            name = resolve_seq_file(self.param_seq_file)
            fig = instructions_figure(channels, title=name.name if name else "")
            other = os.path.join(self.get_working_folder(), "other")
            os.makedirs(other, exist_ok=True)
            with open(os.path.join(other, "sequence.plot"), "wb") as fh:
                pickle.dump(fig, fh)
            plt.close(fig)

            result = ResultItem()
            result.name = "Sequence"
            result.description = "Instructions sent to MaRCoS (values held between updates)"
            result.type = "plot"
            result.primary = True
            result.autoload_viewer = 1
            result.file_path = "other/sequence.plot"
            scan_task.results.insert(0, result)
        except Exception as exc:  # a plot must never fail the scan
            log.warning("Could not plot sequence instructions: %s", exc)

    def run_sequence(self, scan_task) -> bool:
        log.info("Running %s", self.seq_file_path)
        plt.close("all")
        rxd, _ = run_pulseq(
            seq_file=self.seq_file_path,
            rf_center=scan_task.adjustment.rf.larmor_frequency,
            tx_t=1,
            grad_t=10,
            tx_warmup=100,
            shim_x=cfg.SHIM_X,
            shim_y=cfg.SHIM_Y,
            shim_z=cfg.SHIM_Z,
            grad_cal=False,
            save_np=True,
            save_mat=False,
            save_msgs=False,
            case_path=self.get_working_folder(),
            raw_filename="raw",
            plot_instructions=True,   # drawn before the hardware step, so also in simulation
        )
        self._attach_sequence_plot(scan_task)
        if rxd is None or getattr(rxd, "size", 0) == 0:
            log.info("No raw data (hardware simulation or empty acquisition)")
        return True
