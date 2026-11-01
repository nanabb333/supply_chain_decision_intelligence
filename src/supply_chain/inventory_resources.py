"""Bounded process supervisor, exercised only with synthetic subprocesses in C1.

RSS watchdog is sampled, not a kernel-enforced instantaneous memory ceiling.
"""
from dataclasses import dataclass
import os, signal, subprocess, time
import psutil

@dataclass(frozen=True)
class Limits:
    seconds: float = 60.0
    memory_gate: int = 1024**3
    hard_memory_stop: int = 2*1024**3
    poll_seconds: float = .01

    def validate(self):
        import math
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in (self.seconds,self.memory_gate,self.hard_memory_stop,self.poll_seconds)):
            raise ValueError('Positive finite resource limits required')
        if self.memory_gate>self.hard_memory_stop or self.poll_seconds>self.seconds:
            raise ValueError('Invalid gate ordering')
        return self


def violation(elapsed,rss,limits):
    limits.validate()
    if rss>=limits.hard_memory_stop:return 'HARD_MEMORY_STOP'
    if elapsed>=limits.seconds:return 'RUNTIME_GATE'
    if rss>limits.memory_gate:return 'MEMORY_GATE'
    return None


def supervise(command, stdout_path, stderr_path, limits=Limits()):
    """No empirical command is supplied or launched by C1. Fresh process; refuse overwrite."""
    limits.validate()
    from pathlib import Path
    out,err=Path(stdout_path),Path(stderr_path)
    if out==err or out.exists() or err.exists():raise FileExistsError('Resource logs must be new separate files')
    started=time.perf_counter();peak=0;reason=None
    with out.open('x') as of,err.open('x') as ef:
        p=subprocess.Popen(command,stdout=of,stderr=ef,start_new_session=True)
        proc=psutil.Process(p.pid)
        try:
            while p.poll() is None:
                try:
                    processes=[proc]+proc.children(recursive=True)
                    rss=sum(x.memory_info().rss for x in processes if x.is_running())
                    peak=max(peak,rss)
                except psutil.NoSuchProcess:pass
                reason=violation(time.perf_counter()-started,peak,limits)
                if reason:
                    try:os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    break
                time.sleep(limits.poll_seconds)
            code=p.wait()
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGKILL);p.wait()
    elapsed=time.perf_counter()-started
    reason=reason or violation(elapsed,peak,limits)
    return dict(status='PASS' if reason is None and code==0 else 'FAIL',stop_reason=reason,
                exit_code=code,elapsed_seconds=elapsed,peak_sampled_tree_rss_bytes=peak,
                rss_semantics='10ms sampled process-tree RSS; short spikes can escape; not an OS hard ceiling')
