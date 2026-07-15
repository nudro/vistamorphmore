"""
TensorBoard paths and SummaryWriter defaults for VMM trainers.

Default layout: ``<output_dir>/tb/<experiment>/`` so you can run::

    tensorboard --logdir /abs/path/to/.../outputs/tb

and see **one run per experiment** name in the UI. (Avoid pointing --logdir at a single
run folder if you want multiple runs listed side-by-side.)

Event files are written **directly** under that run directory, named
``events.out.tfevents.*`` (there is typically **no** subdirectory literally called ``events``).
Point ``tensorboard --logdir`` at ``.../tb`` or at ``.../tb/<experiment>``, not at a
non-existent ``.../events`` folder.
"""

from __future__ import annotations

import glob
import os
import sys

from torch.utils.tensorboard import SummaryWriter


def resolve_tb_log_dir(output_dir: str, experiment: str, tensorboard_dir_override: str | None) -> tuple[str, bool]:
    """
    Returns (absolute_log_dir, is_override). Override = user passed --tensorboard_dir.
    """
    o = (tensorboard_dir_override or "").strip()
    if o:
        return os.path.abspath(os.path.expanduser(o)), True
    return os.path.abspath(os.path.join(output_dir, "tb", experiment)), False


def tb_runs_parent(output_dir: str) -> str:
    return os.path.abspath(os.path.join(output_dir, "tb"))


def create_summary_writer(log_dir: str) -> SummaryWriter:
    """Create a writer; ``log_dir`` is forced absolute so training cwd does not move files."""
    log_dir = os.path.abspath(os.path.expanduser(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir, flush_secs=1, max_queue=32)


def print_tensorboard_cli_help(tb_run_dir: str, output_dir: str, overridden: bool) -> None:
    tb_run_dir = os.path.abspath(tb_run_dir)
    print(f"TensorBoard event dir (this run): {tb_run_dir}", file=sys.stderr)
    print(
        "  (Scalars live in files named events.out.tfevents.* in that folder — not in a subdir named 'events'.)",
        file=sys.stderr,
    )
    if not overridden:
        parent = tb_runs_parent(output_dir)
        print(
            f"TensorBoard — list ALL experiments: tensorboard --logdir {parent!r} --reload_interval 5",
            file=sys.stderr,
        )
    else:
        print(
            f"TensorBoard — this run only: tensorboard --logdir {tb_run_dir!r} --reload_interval 5",
            file=sys.stderr,
        )
    print(
        "  (--reload_interval 5 helps while training is still writing; omit if loading finished logs only.)",
        file=sys.stderr,
    )


def write_startup_scalars(writer: SummaryWriter, *, learning_rate: float) -> None:
    """Ensure an event file exists and TensorBoard shows the run immediately."""
    writer.add_scalar("meta/run_started", 1.0, 0)
    writer.add_scalar("meta/learning_rate", learning_rate, 0)
    writer.flush()
    _emit_tb_event_file_diagnostic(writer.get_logdir())


def _emit_tb_event_file_diagnostic(log_dir: str) -> None:
    log_dir = os.path.abspath(log_dir)
    try:
        pattern = os.path.join(log_dir, "events.out.tfevents.*")
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(
                f"TensorBoard warning: no events.out.tfevents.* files under {log_dir!r} after startup.",
                file=sys.stderr,
            )
        else:
            print(
                f"TensorBoard: wrote {len(matches)} event file(s); e.g. {matches[0]!r}",
                file=sys.stderr,
            )
    except OSError as exc:
        print(f"TensorBoard diagnostic: could not glob {log_dir!r}: {exc}", file=sys.stderr)
