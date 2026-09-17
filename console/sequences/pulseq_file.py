"""Play a Pulseq .seq file chosen in the Imaging Console.

Browse in the SEQUENCE tab (or run ``scripts/load_seq.py``) to check a file and
copy it into ``<MRI4ALL_BASE>/seq_library/``. This sequence plays whichever
library file ``param_seq_file`` names; the default ``latest`` is the most
recently loaded one.
"""

import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def library_filename(name: str) -> str:
    """Safe ``*.seq`` name that cannot escape the library directory."""
    stem = Path(name or "sequence").name
    if not stem.lower().endswith(".seq"):
        stem = f"{Path(stem).stem or 'sequence'}.seq"
    return stem


def stage_seq_file(source: Path, name: Optional[str] = None) -> Path:
    """Copy ``source`` into the session library and mark it as latest."""
    library = library_dir()
    library.mkdir(parents=True, exist_ok=True)
    dest = library / library_filename(name or source.name)
    shutil.copyfile(source, dest)
    (library / LATEST_POINTER).write_text(dest.name, encoding="utf-8")
    return dest


def check_seq_file(seq_path: Path, plot_path: Optional[Path] = None) -> List[str]:
    """Return problems; empty means the session can play the file."""
    import numpy as np
    import pypulseq as pp

    import external.seq.adjustments_acq.config as cfg
    from external.flocra_pulseq.interpreter_pp import seq2flocra

    problems: List[str] = []
    seq = pp.Sequence()
    try:
        seq.read(str(seq_path))
    except Exception as exc:
        return [f"could not read the file: {exc}"]

    ok, report = seq.check_timing()
    if not ok:
        log.warning("Timing checker flagged %s: %s", seq_path.name, "; ".join(str(x) for x in report))

    has_adc = any(getattr(seq.get_block(i), "adc", None) is not None for i in seq.block_events)
    if not has_adc:
        return problems + ["no ADC event; the interpreter needs at least one"]

    rf_max, g_max = float(cfg.RF_MAX), (float(cfg.GX_MAX), float(cfg.GY_MAX), float(cfg.GZ_MAX))
    system = pp.Opts(
        max_grad=max(g_max),
        grad_unit="Hz/m",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=20e-6,
        rf_raster_time=1e-6,
        grad_raster_time=10e-6,
        block_duration_raster=1e-6,
    )
    psi = seq2flocra(
        center_freq=float(cfg.LARMOR_FREQ) * 1e6,
        rf_amp_max=rf_max,
        system=system,
        clk_freq=122.88,
        gx_max=g_max[0],
        gy_max=g_max[1],
        gz_max=g_max[2],
    )
    try:
        psi.load_seqfile(str(seq_path))
        psi.block_events_to_amps_times()
    except Exception as exc:
        return problems + [f"interpreter failed: {type(exc).__name__}: {exc}"]

    for channel, (_, amps) in psi._flo_dict.items():
        peak = float(np.abs(np.asarray(amps)).max())
        if channel != "tx_gate" and channel != "rx0_en" and peak > 1.0:
            problems.append(f"{channel} peaks at {peak:.2f}x DAC full scale; it would clip")

    duration = seq.duration()[0]
    log.info(
        "%s: %s blocks, %.3f s, calibration %.4f MHz / RF_MAX %.0f Hz",
        seq_path.name,
        len(seq.block_events),
        duration,
        float(cfg.LARMOR_FREQ),
        rf_max,
    )

    if plot_path is not None:
        instructions_figure(psi._flo_dict, title=seq_path.name).savefig(plot_path, dpi=110)
    return problems


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
        description="Choose a Pulseq .seq file, or leave as latest for the last one loaded",
        widget="file",
        accept=".seq",
    )

    @classmethod
    def get_readable_name(cls) -> str:
        return "Run .seq file"

    @classmethod
    def get_description(cls) -> str:
        return "Plays a Pulseq .seq file chosen in the Imaging Console"

    def validate_parameters(self, scan_task) -> bool:
        if resolve_seq_file(self.param_seq_file) is None:
            self.problem_list.append(
                f"No '{self.param_seq_file}' in {library_dir()}. "
                "Choose a .seq file in the SEQUENCE tab."
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
