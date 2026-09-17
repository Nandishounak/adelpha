#!/usr/bin/env python3
"""Check a Pulseq .seq file and make it available to scan in Adelpha.

    python scripts/load_seq.py path/to/sequence.seq

Checks run with the same interpreter and PyPulseq the Adelpha session uses, so
a file that passes here is one the scanner can interpret. Whatever Python you
start it with, it re-runs itself under the session's interpreter
(runtime/python/.venv). On success the file is copied into the session's seq
library and marked as the latest; in Adelpha, add "Run .seq file" to the queue
and scan.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONSOLE = REPO / "console"
SESSION_VENV = REPO / "runtime" / "python" / ".venv"
SESSION_PYTHON = SESSION_VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_session_python():
    """Re-run under the Adelpha session's interpreter unless already there."""
    if Path(sys.prefix).resolve() == SESSION_VENV.resolve():
        return
    if not SESSION_PYTHON.exists():
        sys.exit(f"Adelpha's Python runtime is missing ({SESSION_PYTHON}). Run `make install` first.")
    # Compare prefixes, not executables: venv pythons are symlinks to the base interpreter.
    raise SystemExit(subprocess.call([str(SESSION_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]]))
LIBRARY_DIRNAME = "seq_library"  # keep in sync with console/sequences/pulseq_file.py
LATEST_POINTER = "LATEST"


def default_base() -> Path:
    """MRI4ALL base of the desktop app (Tauri sets ADELPHA_DATA_DIR to this)."""
    if os.environ.get("MRI4ALL_BASE"):
        return Path(os.environ["MRI4ALL_BASE"])
    home = Path.home()
    if sys.platform == "darwin":
        data = home / "Library" / "Application Support" / "org.adelpha.digital-twin-ui"
    elif sys.platform == "win32":
        data = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming")) / "org.adelpha.digital-twin-ui"
    else:
        data = Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share")) / "org.adelpha.digital-twin-ui"
    return data / "mri4all"


def check(seq_path: Path, plot_path: Path | None = None) -> list[str]:
    """Return a list of problems; empty means the session can play the file."""
    # Match the session: console/ first, so the vendored PyPulseq is the one used.
    sys.path.insert(0, str(CONSOLE))
    import logging

    logging.disable(logging.WARNING)  # the sequence registry is noisy on import
    import numpy as np
    import pypulseq as pp

    import external.seq.adjustments_acq.config as cfg
    from external.flocra_pulseq.interpreter_pp import seq2flocra

    problems = []

    seq = pp.Sequence()
    try:
        seq.read(str(seq_path))
    except Exception as exc:
        return [f"could not read the file: {exc}"]

    ok, report = seq.check_timing()
    if not ok:
        problems += [f"timing: {line}" for line in report]

    # The vendored get_block omits absent events rather than setting them to None.
    has_adc = any(getattr(seq.get_block(i), "adc", None) is not None for i in seq.block_events)
    if not has_adc:
        # The interpreter fails on RF-only sequences.
        return problems + ["no ADC event; the interpreter needs at least one"]

    rf_max, g_max = float(cfg.RF_MAX), (float(cfg.GX_MAX), float(cfg.GY_MAX), float(cfg.GZ_MAX))
    system = pp.Opts(
        max_grad=max(g_max), grad_unit="Hz/m",
        rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=20e-6,
        rf_raster_time=1e-6, grad_raster_time=10e-6, block_duration_raster=1e-6,
    )
    psi = seq2flocra(
        center_freq=float(cfg.LARMOR_FREQ) * 1e6, rf_amp_max=rf_max, system=system,
        clk_freq=122.88, gx_max=g_max[0], gy_max=g_max[1], gz_max=g_max[2],
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
    print(f"  {len(seq.block_events)} blocks, {duration:.3f} s, "
          f"calibration {float(cfg.LARMOR_FREQ):.4f} MHz / RF_MAX {rf_max:.0f} Hz")

    if plot_path is not None:
        # Same figure the "Run .seq file" scan attaches in Adelpha.
        from sequences.pulseq_file import instructions_figure

        instructions_figure(psi._flo_dict, title=seq_path.name).savefig(plot_path, dpi=110)
        print(f"  plot: {plot_path}")
    return problems


def main() -> int:
    ensure_session_python()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seq", type=Path, help="the .seq file to load")
    parser.add_argument("--name", help="name in the library (default: the file name)")
    parser.add_argument("--base", type=Path, help="MRI4ALL base dir (default: the desktop app's)")
    parser.add_argument("--check-only", action="store_true", help="check, but do not load")
    parser.add_argument("--plot", type=Path, metavar="PNG",
                        help="save the MaRCoS instructions as a PNG (what the scanner will play)")
    args = parser.parse_args()

    seq_path = args.seq.resolve()
    if not seq_path.is_file():
        print(f"not found: {seq_path}")
        return 1

    base = args.base or default_base()
    os.environ["MRI4ALL_BASE"] = str(base)  # config and logs resolve against this

    print(f"Checking {seq_path.name}")
    problems = check(seq_path, args.plot.resolve() if args.plot else None)
    if problems:
        print("FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("  OK")

    if args.check_only:
        return 0

    name = Path(args.name or seq_path.name).stem + ".seq"
    library = base / LIBRARY_DIRNAME
    library.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(seq_path, library / name)
    (library / LATEST_POINTER).write_text(name, encoding="utf-8")

    print(f"Loaded as {library / name}")
    print('In Adelpha: Imaging Console -> add "Run .seq file" -> scan.')
    print(f'(Sequence file = "latest", or "{name}" to pick it explicitly later.)')
    return 0


if __name__ == "__main__":
    sys.exit(main())
