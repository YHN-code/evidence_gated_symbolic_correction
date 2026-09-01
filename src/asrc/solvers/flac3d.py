from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class FLAC3DError(RuntimeError):
    """Raised when FLAC3D cannot complete a requested data file."""


class FLAC3DProcessCleanupError(FLAC3DError):
    """Raised when a failed FLAC3D child cannot be confirmed as terminated."""


@dataclass(frozen=True)
class FLAC3DRunResult:
    run_dir: Path
    transcript_path: Path
    elapsed_seconds: float
    exit_code: int | None


_ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_SOLVE_PROGRESS = re.compile(
    r"^\s*\d+\s+\d+\s+[+\-0-9.eE]+\s+[+\-0-9.eE]+\s+\d{2}:\d{2}:\d{2}:\d{2}\s*$"
)
_FLAC3D_ERROR_LINE = re.compile(r"(?m)^\s*\*{3}\s+\S")


def clean_flac3d_transcript(text: str) -> str:
    """Remove terminal control codes and repetitive cycle-status rows."""
    text = _ANSI_ESCAPE.sub("", text)
    rendered: list[str] = []
    for character in text:
        if character == "\b":
            if rendered and rendered[-1] != "\n":
                rendered.pop()
            continue
        rendered.append(character)
    normalized = "".join(rendered).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.splitlines()]
    lines = [line for line in lines if not _SOLVE_PROGRESS.match(line)]

    compact: list[str] = []
    for line in lines:
        if not line and compact and not compact[-1]:
            continue
        compact.append(line)
    return "\n".join(compact).strip() + "\n"


def contains_flac3d_error(text: str) -> bool:
    """Detect FLAC3D error records without matching save-file banner rules."""
    return bool(_FLAC3D_ERROR_LINE.search(clean_flac3d_transcript(text)))


def _program_call_path(data_path: Path, output_dir: Path) -> str:
    """Prefer a short path when FLAC3D is launched in the data-file directory."""
    try:
        relative = data_path.relative_to(output_dir)
    except ValueError:
        selected = data_path
    else:
        selected = relative
    return selected.as_posix().replace("'", "''")


def _reader(process: object, output: queue.Queue[str | None]) -> None:
    try:
        while process.isalive():
            try:
                chunk = process.read(4096)
            except EOFError:
                break
            if chunk:
                output.put(chunk)
        while not process.eof():
            try:
                chunk = process.read(4096)
            except (EOFError, OSError):
                break
            if chunk:
                output.put(chunk)
    finally:
        output.put(None)


def _wait_for(
    output: queue.Queue[str | None],
    predicate: Callable[[str], bool],
    timeout_seconds: float,
    on_chunk: Callable[[str], None],
) -> str:
    deadline = time.monotonic() + timeout_seconds
    collected = ""
    while time.monotonic() < deadline:
        # Some FLAC3D console prompts remain in a short winpty tail. Allow a
        # caller-provided artifact predicate to complete the wait even when no
        # additional console block is available.
        if predicate(collected):
            return collected
        try:
            chunk = output.get(timeout=min(1.0, max(0.01, deadline - time.monotonic())))
        except queue.Empty:
            continue
        if chunk is None:
            raise FLAC3DError("FLAC3D exited before the expected console state was reached.")
        collected += chunk
        on_chunk(chunk)
        if predicate(collected):
            return collected
    raise FLAC3DError(f"FLAC3D timed out after {timeout_seconds:.0f} seconds.")


def _drain_until_idle(
    output: queue.Queue[str | None],
    on_chunk: Callable[[str], None],
    *,
    idle_seconds: float = 0.75,
) -> None:
    while True:
        try:
            chunk = output.get(timeout=idle_seconds)
        except queue.Empty:
            return
        if chunk is None:
            raise FLAC3DError("FLAC3D exited during console initialization.")
        on_chunk(chunk)


def _wait_for_process_exit(process: object, timeout_seconds: float) -> bool:
    """Wait for a PTY child without relying on its potentially blocking wait()."""
    deadline = time.monotonic() + timeout_seconds
    while process.isalive() and time.monotonic() < deadline:
        time.sleep(0.05)
    return not process.isalive()


def _terminate_process_safely(
    process: object,
    *,
    timeout_seconds: float = 5.0,
) -> str | None:
    """Terminate a PTY child while tolerating Windows process-exit races."""
    try:
        if not process.isalive():
            return None
    except OSError as exc:
        return f"could not inspect the FLAC3D process: {exc}"

    try:
        process.terminate(force=True)
    except OSError as exc:
        # On Windows, winpty may report access denied after the child has
        # already exited. Confirm its state before treating cleanup as failed.
        try:
            if _wait_for_process_exit(process, min(timeout_seconds, 2.0)):
                return None
        except OSError:
            pass
        return f"could not terminate the FLAC3D process: {exc}"

    try:
        if _wait_for_process_exit(process, timeout_seconds):
            return None
    except OSError as exc:
        return f"could not confirm FLAC3D process exit: {exc}"
    return f"FLAC3D remained alive for {timeout_seconds:g} seconds after termination"


def run_flac3d_data_file(
    executable: str | Path,
    data_file: str | Path,
    run_dir: str | Path,
    *,
    timeout_seconds: float = 900.0,
    startup_timeout_seconds: float = 120.0,
    verbose: bool = False,
    progress_only: bool = False,
    completion_check: Callable[[], bool] | None = None,
) -> FLAC3DRunResult:
    try:
        from winpty import PtyProcess
    except ImportError as exc:
        raise FLAC3DError(
            "FLAC3D automation requires pywinpty. Install it with "
            "`python -m pip install -e .[flac3d]`."
        ) from exc

    executable_path = Path(executable).expanduser().resolve()
    data_path = Path(data_file).expanduser().resolve()
    output_dir = Path(run_dir).expanduser().resolve()
    if not executable_path.is_file():
        raise FLAC3DError(f"FLAC3D console executable was not found: {executable_path}")
    if not data_path.is_file():
        raise FLAC3DError(f"FLAC3D data file was not found: {data_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = output_dir / "flac3d_console.log"
    transcript_path.write_text("", encoding="utf-8")
    transcript_parts: list[str] = []
    progress_buffer = ""

    def record(chunk: str) -> None:
        nonlocal progress_buffer
        transcript_parts.append(chunk)
        # Keep a crash-readable transcript even if the parent Python process is
        # interrupted before the final cleanup pass can normalize the log.
        with transcript_path.open("a", encoding="utf-8") as stream:
            stream.write(chunk)
            stream.flush()
        if verbose and not progress_only:
            print(chunk, end="", flush=True)
        elif verbose:
            normalized = _ANSI_ESCAPE.sub("", chunk).replace("\r", "\n")
            progress_buffer += normalized
            lines = progress_buffer.split("\n")
            progress_buffer = lines.pop()
            for line in lines:
                cleaned = line.strip()
                if cleaned.startswith(("ASRC_STAGE", "ASRC_RESULT", "--- Limit ratio")):
                    print(cleaned, flush=True)

    start = time.monotonic()
    process = PtyProcess.spawn(
        [str(executable_path)],
        cwd=str(output_dir),
        dimensions=(40, 160),
    )
    output: queue.Queue[str | None] = queue.Queue()
    thread = threading.Thread(target=_reader, args=(process, output), daemon=True)
    thread.start()

    try:
        _wait_for(
            output,
            lambda text: "flac3d>" in text.lower(),
            startup_timeout_seconds,
            record,
        )
        _drain_until_idle(output, record)
        command_path = _program_call_path(data_path, output_dir)
        process.write(f"program call '{command_path}'\r")

        def case_completed(text: str) -> bool:
            recent_console = text[-16384:]
            recent_error = contains_flac3d_error(recent_console)
            console_completed = bool(
                re.search(r"flac3d>\s*$", text, flags=re.IGNORECASE)
            ) and ("ASRC_RESULT" in text or recent_error)
            artifact_completed = bool(completion_check and completion_check())
            return console_completed or artifact_completed or recent_error

        completed_text = _wait_for(
            output,
            case_completed,
            timeout_seconds,
            record,
        )
        if contains_flac3d_error(completed_text):
            raise FLAC3DError("FLAC3D reported an error; inspect flac3d_console.log.")
        artifact_completed = bool(completion_check and completion_check())
        if "ASRC_RESULT" not in completed_text and not artifact_completed:
            raise FLAC3DError(
                "FLAC3D returned to its prompt before ASRC_RESULT; "
                "inspect flac3d_console.log."
            )
        process.write("program quit\r")
        if not _wait_for_process_exit(process, 10.0):
            # The console occasionally returns to its prompt but ignores quit.
            # At this point the completion artifact or ASRC_RESULT is complete.
            cleanup_error = _terminate_process_safely(process)
            if cleanup_error:
                raise FLAC3DProcessCleanupError(
                    "FLAC3D completed the case but its console could not be closed: "
                    f"{cleanup_error}."
                )
        thread.join(timeout=1.0)
        while True:
            try:
                chunk = output.get_nowait()
            except queue.Empty:
                break
            if chunk is None:
                break
            record(chunk)
    except BaseException as exc:
        cleanup_error = _terminate_process_safely(process)
        if cleanup_error:
            raise FLAC3DProcessCleanupError(
                f"{exc} Cleanup also failed: {cleanup_error}."
            ) from exc
        raise
    finally:
        transcript_path.write_text(
            clean_flac3d_transcript("".join(transcript_parts)),
            encoding="utf-8",
        )

    elapsed = time.monotonic() - start
    return FLAC3DRunResult(
        run_dir=output_dir,
        transcript_path=transcript_path,
        elapsed_seconds=elapsed,
        exit_code=process.exitstatus,
    )
