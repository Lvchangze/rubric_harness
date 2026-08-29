"""Subprocess sandbox for model-authored Python.

The code executed here is written by the model, so it is treated as untrusted
input rather than as part of the program. Isolation is layered:

* a **separate process**, so a segfault, ``sys.exit`` or runaway allocation
  cannot take the generator down;
* ``-I`` (isolated mode), so the harness's own ``sys.path``, ``PYTHONPATH`` and
  user site-packages are not visible and the model cannot import project code;
* **rlimits** on CPU seconds, address space and file size;
* a **wall-clock timeout** enforced by the parent, which kills the process group
  — an rlimit alone does not stop a process blocked in a syscall;
* an import hook that refuses networking and process-spawning modules;
* a scratch cwd under ``/tmp`` that is removed afterwards.

This is defence in depth against accidents, not a claim of security against a
determined adversary sharing the host. It is proportionate: the "adversary" is a
model asked to check arithmetic.
"""

from __future__ import annotations

import asyncio
import json
import os
import resource
import shutil
import signal
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

__all__ = ["SandboxResult", "run_python"]

#: Modules the sandboxed process may not import.
#:
#: Deliberately narrow. Networking is blocked because a tool that can reach the
#: internet stops being reproducible; process spawning is blocked because the
#: rlimits bind this process, not children it forks. Everything else stays
#: importable — an earlier, broader list included ``pickle``, which numpy pulls
#: in on import, so blocking it disabled the numeric library the tool exists to
#: provide. Breaking the payload to harden against a threat that is not present
#: (the code's author is a model checking arithmetic, and the process has no
#: credentials) is a bad trade.
#: Whole packages that exist only to talk to the network or start processes.
_BLOCKED_ROOTS = (
    "socket", "ssl", "ftplib", "telnetlib", "smtplib", "poplib", "imaplib",
    "requests", "httpx", "urllib3", "xmlrpc", "asyncio",
    "multiprocessing", "pty", "webbrowser",
)

#: Importable, but with the functions that start a process replaced. ``sympy``
#: imports ``subprocess`` at load time (its printing module can shell out to a
#: LaTeX previewer), so refusing the import removes symbolic maths altogether.
#: Neutering the entry points keeps the capability closed without breaking the
#: library that only wanted the module object.
_NEUTERED = {
    "subprocess": ("Popen", "run", "call", "check_call", "check_output",
                   "getoutput", "getstatusoutput"),
    "os": ("system", "popen", "fork", "forkpty", "execv", "execve", "execvp",
           "execvpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "posix_spawn"),
}

#: Individual transport submodules inside packages that are otherwise fine.
#: ``urllib.parse`` is a string parser and sympy imports it at load time, so
#: blocking the ``urllib`` root disabled sympy entirely; only ``urllib.request``
#: and friends actually open a connection.
_BLOCKED_PREFIXES = (
    "urllib.request", "urllib.error", "urllib.response",
    "http.client", "http.server", "http.cookiejar",
)

#: Preloaded so the model does not spend a turn importing them.
_PRELUDE = """\
import builtins, math, cmath, statistics, fractions, decimal, itertools, functools, re, json
from fractions import Fraction
from decimal import Decimal
try:
    import numpy
    import numpy as np
except Exception:
    numpy = np = None
try:
    import sympy
    from sympy import (
        symbols, Symbol, simplify, expand, factor, solve, nsolve, Eq, N, sqrt,
        pi, E, I, oo, diff, integrate, limit, series, Rational, S, sympify,
    )
except Exception:
    sympy = None
"""

_RUNNER_TEMPLATE = '''\
import sys, os, json, io, contextlib, traceback

_BLOCKED_ROOTS = set({blocked_roots!r})
_BLOCKED_PREFIXES = tuple({blocked_prefixes!r})
_NEUTERED = {neutered!r}
_real_import = __import__

def _deny(*_a, **_k):
    raise RuntimeError("starting processes is not available in this sandbox")

for _mod_name, _attrs in _NEUTERED.items():
    try:
        _mod = _real_import(_mod_name)
    except Exception:
        continue
    for _attr in _attrs:
        if hasattr(_mod, _attr):
            try:
                setattr(_mod, _attr, _deny)
            except Exception:
                pass

def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    blocked = name.split(".")[0] in _BLOCKED_ROOTS or any(
        name == p or name.startswith(p + ".") for p in _BLOCKED_PREFIXES
    )
    if blocked:
        raise ImportError(
            "module %r is not available in this sandbox (no network, no subprocesses)" % name
        )
    return _real_import(name, globals, locals, fromlist, level)

import builtins as _builtins
_builtins.__import__ = _guarded_import
_builtins.input = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stdin is not available"))
_builtins.open = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("file access is not available"))

_env = {{}}
_stdout = io.StringIO()
_payload = {{"ok": True, "stdout": "", "value": None, "error": None}}

try:
    with contextlib.redirect_stdout(_stdout), contextlib.redirect_stderr(_stdout):
        exec(compile({prelude!r}, "<prelude>", "exec"), _env)
        _src = {code!r}
        try:
            # A bare expression is evaluated and its value reported, so `2 + 2`
            # is useful without the model remembering to print().
            _compiled = compile(_src, "<model>", "eval")
        except SyntaxError:
            exec(compile(_src, "<model>", "exec"), _env)
            if "result" in _env:
                _payload["value"] = repr(_env["result"])
        else:
            _value = eval(_compiled, _env)
            if _value is not None:
                _payload["value"] = repr(_value)
except BaseException as exc:
    _payload["ok"] = False
    _payload["error"] = "%s: %s" % (type(exc).__name__, exc)
    _payload["traceback"] = traceback.format_exc(limit=4)[-2000:]

_payload["stdout"] = _stdout.getvalue()
sys.__stdout__.write("\\n<<<SANDBOX_JSON>>>" + json.dumps(_payload, default=str))
'''


@dataclass
class SandboxResult:
    ok: bool
    stdout: str = ""
    value: str | None = None
    error: str | None = None
    traceback: str = ""
    timed_out: bool = False
    seconds: float = 0.0
    exit_code: int | None = None


def _limits(cpu_seconds: int, memory_mb: int):
    def apply() -> None:
        os.setsid()  # own process group, so a timeout kill reaches grandchildren
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        nbytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        # No RLIMIT_NPROC: it is per-*user*, not per-process, so a low value is
        # both ineffective (other processes of this user already count against
        # it) and destructive — OpenBLAS opens one thread per core at numpy
        # import and aborts the interpreter when pthread_create fails. Thread
        # count is capped through the environment instead.

    return apply


#: Numeric libraries default to one thread per core and this process only needs
#: a scratch calculator. Capping them keeps memory inside RLIMIT_AS and avoids
#: OpenBLAS aborting at import when it cannot spawn its pool.
_THREAD_CAPS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


async def run_python(
    code: str,
    *,
    timeout_s: float = 20.0,
    cpu_seconds: int = 15,
    memory_mb: int = 1024,
    max_output_chars: int = 4000,
) -> SandboxResult:
    """Execute ``code`` in an isolated interpreter and return what it produced."""
    if not (code or "").strip():
        return SandboxResult(ok=False, error="no code supplied")

    loop = asyncio.get_running_loop()
    started = loop.time()
    workdir = tempfile.mkdtemp(prefix="rubric_sandbox_")
    runner = os.path.join(workdir, "runner.py")
    try:
        with open(runner, "w", encoding="utf-8") as fh:
            fh.write(
                _RUNNER_TEMPLATE.format(
                    blocked_roots=sorted(_BLOCKED_ROOTS),
                    blocked_prefixes=sorted(_BLOCKED_PREFIXES),
                    neutered={k: list(v) for k, v in _NEUTERED.items()},
                    prelude=_PRELUDE,
                    code=code,
                )
            )

        # -I isolates from PYTHONPATH/user-site; an emptied env keeps it that way.
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", runner,
            cwd=workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=_limits(cpu_seconds, memory_mb),
            env={"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir,
                 "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg", **_THREAD_CAPS},
        )
        try:
            raw_out, raw_err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            _kill_group(proc)
            await proc.wait()
            return SandboxResult(
                ok=False,
                error=f"execution exceeded {timeout_s:.0f}s and was terminated",
                timed_out=True,
                seconds=loop.time() - started,
            )
        except asyncio.CancelledError:
            # An outer timeout (the registry's) cancels us. Without this the
            # subprocess survives its parent and keeps burning a core until the
            # CPU rlimit fires.
            _kill_group(proc)
            raise

        stdout = raw_out.decode("utf-8", "replace")
        stderr = raw_err.decode("utf-8", "replace")
        payload, tail = _split_payload(stdout)
        elapsed = loop.time() - started

        if payload is None:
            detail = (stderr or tail).strip()[-600:]
            # An rlimit kill leaves no payload; say which limit rather than
            # surfacing a bare non-zero exit code.
            if proc.returncode and proc.returncode < 0:
                sig = -proc.returncode
                hint = {
                    signal.SIGXCPU: "CPU time limit exceeded",
                    signal.SIGKILL: "killed (most likely the memory limit)",
                    signal.SIGSEGV: "segmentation fault",
                }.get(sig, f"terminated by signal {sig}")
                detail = f"{hint}. {detail}".strip()
            return SandboxResult(
                ok=False,
                error=detail or "the sandbox produced no output",
                stdout=tail[:max_output_chars],
                seconds=elapsed,
                exit_code=proc.returncode,
            )

        out = str(payload.get("stdout") or "")
        if len(out) > max_output_chars:
            out = out[:max_output_chars] + f"\n…[truncated, {len(out) - max_output_chars} chars]"
        return SandboxResult(
            ok=bool(payload.get("ok")),
            stdout=out,
            value=payload.get("value"),
            error=payload.get("error"),
            traceback=str(payload.get("traceback") or ""),
            seconds=elapsed,
            exit_code=proc.returncode,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _kill_group(proc: Any) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _split_payload(stdout: str) -> tuple[dict[str, Any] | None, str]:
    marker = "<<<SANDBOX_JSON>>>"
    idx = stdout.rfind(marker)
    if idx < 0:
        return None, stdout
    head = stdout[:idx]
    try:
        return json.loads(stdout[idx + len(marker):]), head
    except json.JSONDecodeError:
        return None, stdout
