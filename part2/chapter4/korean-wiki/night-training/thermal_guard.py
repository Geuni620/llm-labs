"""Passive macOS thermal/deadline watchdog; never writes a checkpoint.

NSProcessInfo reports OS thermal pressure, not CPU degrees or fire detection.
Apple documents that unknown/unsupported thermal hardware can always report
nominal: a successful API read cannot establish hardware sensor support.
https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/RespondToThermalStateChanges.html

The training owner must stop every compute/validation/generation phase when
should_stop becomes true. mark_stopped acknowledges all main work, including
pending checkpoint I/O, has ended; immediately close before returning. Hard exits retain only checkpoints
the owner already completed atomically; this module never checkpoints.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import re
import subprocess
import sys
import threading
import time


THERMAL_EXIT_CODE = 86
DEADLINE_EXIT_CODE = 87
STOP_TIMEOUT_EXIT_CODE = 88
STATE_NAMES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}
LIMITATION = "OS thermal pressure only; not degrees/fire detection. Unknown or unsupported hardware may always report nominal."


def utc_now():
    return datetime.now(timezone.utc)


def _aware(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("Absolute deadlines and clocks must use timezone-aware datetime values")
    return value


@dataclass(frozen=True)
class ErrorInfo:
    source: str
    kind: str
    message: str
    unsupported: bool = False


@dataclass(frozen=True)
class ThermalSample:
    sampled_at: datetime
    state: int | None
    supported: bool
    error: ErrorInfo | None = None

    @property
    def ok(self):
        return self.supported and self.error is None and type(self.state) is int and self.state in STATE_NAMES

    @property
    def name(self):
        return STATE_NAMES.get(self.state, "unknown")


@dataclass(frozen=True)
class PowerSample:
    sampled_at: datetime
    on_battery: bool | None
    percent: int | None
    error: ErrorInfo | None = None

    @property
    def ok(self):
        return (self.error is None and type(self.on_battery) is bool
                and (self.percent is None and not self.on_battery
                     or type(self.percent) is int and 0 <= self.percent <= 100))


def _load_thermal_getter():
    """Use distinct objc_msgSend prototypes for object, BOOL and NSInteger."""
    foundation = ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.argtypes, objc.objc_getClass.restype = [ctypes.c_char_p], ctypes.c_void_p
    objc.sel_registerName.argtypes, objc.sel_registerName.restype = [ctypes.c_char_p], ctypes.c_void_p
    address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
    object_message = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(address)
    integer_message = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(address)
    responds = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(address)
    objc.objc_autoreleasePoolPush.argtypes, objc.objc_autoreleasePoolPush.restype = [], ctypes.c_void_p
    objc.objc_autoreleasePoolPop.argtypes, objc.objc_autoreleasePoolPop.restype = [ctypes.c_void_p], None
    klass = objc.objc_getClass(b"NSProcessInfo")
    if not klass:
        raise RuntimeError("NSProcessInfo class is unavailable")
    pool = objc.objc_autoreleasePoolPush()
    try:
        process = object_message(klass, objc.sel_registerName(b"processInfo"))
        selector = objc.sel_registerName(b"thermalState")
        if not process or not responds(process, objc.sel_registerName(b"respondsToSelector:"), selector):
            raise NotImplementedError("NSProcessInfo.thermalState is unavailable")
    finally:
        objc.objc_autoreleasePoolPop(pool)

    def read():
        # Capture both library handles so function pointers remain valid.
        _ = foundation, objc
        pool = objc.objc_autoreleasePoolPush()
        try:
            return int(integer_message(process, selector))
        finally:
            objc.objc_autoreleasePoolPop(pool)
    return read


class MacOSThermalReader:
    """Lazy, passive reader. supported means API available, not sensor proven."""
    def __init__(self, *, clock=utc_now, platform=None, loader=None):
        self.clock = clock
        self.platform = sys.platform if platform is None else platform
        self.loader = _load_thermal_getter if loader is None else loader
        self._getter = None
        self._lock = threading.Lock()

    def sample_now(self):
        at = _aware(self.clock())
        if self.platform != "darwin":
            return ThermalSample(at, None, False, ErrorInfo("thermal", "unsupported_platform", self.platform, True))
        try:
            with self._lock:
                if self._getter is None:
                    self._getter = self.loader()
                state = self._getter()
            if type(state) is not int or state not in STATE_NAMES:
                return ThermalSample(at, state, True, ErrorInfo("thermal", "invalid_state", repr(state)))
            return ThermalSample(at, state, True)
        except Exception as error:
            return ThermalSample(at, None, False, ErrorInfo("thermal", type(error).__name__, str(error), isinstance(error, NotImplementedError)))


_DEFAULT_READER = MacOSThermalReader()


def sample_now():
    return _DEFAULT_READER.sample_now()


def parse_power_sample(text, *, sampled_at=None):
    at = utc_now() if sampled_at is None else _aware(sampled_at)
    source = re.search(r"Now drawing from ['\"](AC Power|Battery Power)['\"]", text)
    if source is None:
        return PowerSample(at, None, None, ErrorInfo("power", "parse_error", "pmset power source is unrecognized"))
    on_battery = source.group(1) == "Battery Power"
    percentages = [int(value) for value in re.findall(r"\b(\d{1,3})\s*%;", text)]
    percent = min(percentages) if percentages else None
    if (percent is None and on_battery) or (percent is not None and not 0 <= percent <= 100):
        return PowerSample(at, on_battery, percent, ErrorInfo("power", "parse_error", "Battery percentage is unavailable/invalid"))
    return PowerSample(at, on_battery, percent)


def read_power_sample(*, clock=utc_now, run=None):
    at = _aware(clock())
    try:
        result = (subprocess.run if run is None else run)(
            ["/usr/bin/pmset", "-g", "batt"], capture_output=True, text=True, timeout=2, check=False)
        if result.returncode:
            raise RuntimeError(f"pmset exited {result.returncode}: {result.stderr.strip()}")
        return parse_power_sample(result.stdout, sampled_at=at)
    except Exception as error:
        return PowerSample(at, None, None, ErrorInfo("power", type(error).__name__, str(error)))


class NightGuard:
    """Independent polling, heartbeat and bounded stop-acknowledgment timers.

    A complete sample must finish its readers and logger. Default freshness is
    20s; stale/serious/low-battery/error stops allow at most 30s for the owner's
    all-work-stopped acknowledgment. Critical keeps its shorter 5s cutoff.
    emit must be a prompt, thread-safe logger. Timer exits do not call emit
    first, so a blocked logger/reader cannot defer an already armed cutoff.
    Timers/clock/readers/exits are injectable; tests never need OS telemetry.
    """
    def __init__(self, training_stop_at, hard_stop_at, *, clock=utc_now,
                 monotonic=time.monotonic, read_thermal=None, read_power=None,
                 hard_exit=None, emit=None, sample_interval=5.0, power_interval=30.0,
                 critical_grace=5.0, monitor_power=True, timer_factory=threading.Timer,
                 stale_after=20.0, stop_grace=30.0):
        self.training_stop_at, self.hard_stop_at = _aware(training_stop_at), _aware(hard_stop_at)
        if self.hard_stop_at <= self.training_stop_at:
            raise ValueError("Hard deadline must follow the absolute training deadline")
        if not (0 < sample_interval <= 5 and 0 < critical_grace <= 5 and 0 < power_interval <= 30):
            raise ValueError("Sampling/critical grace must be <=5s and power sampling <=30s")
        if not (15 <= stale_after <= 30 and 0 < stop_grace <= 30):
            raise ValueError("Heartbeat stale limit must be 15..30s; protective-stop grace must be <=30s")
        self.clock, self.monotonic = clock, monotonic
        self.read_thermal = sample_now if read_thermal is None else read_thermal
        self.read_power = read_power_sample if read_power is None else read_power
        self.hard_exit = os._exit if hard_exit is None else hard_exit
        self.emit = emit
        self.sample_interval, self.power_interval, self.critical_grace = sample_interval, power_interval, critical_grace
        self.monitor_power, self.timer_factory = monitor_power, timer_factory
        self.stale_after, self.stop_grace = stale_after, stop_grace
        self.event, self.ready_event, self.initial_sample_event = threading.Event(), threading.Event(), threading.Event()
        self._closed, self._stopped = threading.Event(), threading.Event()
        self._lock, self._poll_lock = threading.RLock(), threading.Lock()
        self._thread = self._critical_timer = self._hard_timer = self._training_timer = None
        self._heartbeat_timer = self._stop_timer = None
        self._started = False
        self._critical_due = self._next_power = None
        self._heartbeat_anchor = self._last_success_mono = self._last_success_at = None
        self._stop_due = self._stop_origin_reason = None
        self._stop_timer_armed = False
        self._reason = None
        self._reasons = []
        self._hard_exit_requested = False
        self.last_thermal = self.last_power = self.last_error = None

    @property
    def should_stop(self):
        return self.event.is_set()

    @property
    def reason(self):
        with self._lock:
            return self._reason

    @property
    def initial_ready(self):
        return self.ready_event.is_set() and not self.should_stop

    def wait_ready(self, timeout=None):
        return self.initial_sample_event.wait(timeout) and self.initial_ready

    def _log(self, kind, **details):
        if self.emit is not None:
            try:
                self.emit({"kind": kind, **details})
            except Exception as error:
                self.last_error = ErrorInfo("emit", type(error).__name__, str(error))
                self._stop("monitor_error", log=False)
                return False
        return True

    def _stop(self, reason, *, log=True):
        priority = {"training_deadline": 10, "low_battery": 20, "thermal_serious": 30,
                    "monitor_error": 40, "thermal_critical": 50, "hard_deadline": 60}
        arm_stop = False
        with self._lock:
            changed = reason not in self._reasons
            if changed:
                self._reasons.append(reason)
            if self._reason is None or priority[reason] > priority[self._reason]:
                self._reason = reason
            self.event.set()
            if (reason in {"thermal_serious", "low_battery", "monitor_error"}
                    and not self._stop_timer_armed and not self._stopped.is_set()
                    and not self._closed.is_set() and not self._hard_exit_requested):
                self._stop_timer_armed = True
                self._stop_origin_reason = reason
                try:
                    self._stop_due = self.monotonic() + self.stop_grace
                except Exception:
                    self._stop_due = None  # The real timer still expires in 30s.
                arm_stop = True
        if arm_stop:
            self._arm_timer("_stop_timer", self.stop_grace, self._stop_timeout)
        if changed and log:
            self._log("stop_requested", reason=reason)

    def _exit_once(self, code, reason):
        with self._lock:
            if self._closed.is_set() or self._hard_exit_requested:
                return
            if reason != "hard_deadline" and self._stopped.is_set():
                return
            self._hard_exit_requested = True
            self._stop(reason, log=False)
        self.hard_exit(code)

    def _arm_timer(self, kind, delay, callback):
        with self._lock:
            if self._closed.is_set() or self._hard_exit_requested:
                return
            if kind in {"_critical_timer", "_heartbeat_timer", "_stop_timer"} and self._stopped.is_set():
                return
            previous = getattr(self, kind)
            if previous is not None:
                previous.cancel()
            timer = self.timer_factory(max(0.0, delay), callback)
            timer.daemon = True
            setattr(self, kind, timer)
            timer.start()

    def _hard_timeout(self):
        if self._closed.is_set():
            return
        try:
            remaining = (self.hard_stop_at - _aware(self.clock())).total_seconds()
        except Exception as error:
            self.last_error = ErrorInfo("clock", type(error).__name__, str(error))
            self._stop("monitor_error", log=False)
            remaining = 0  # Do not postpone the already armed absolute cutoff.
        if remaining > 0:
            self._arm_timer("_hard_timer", remaining, self._hard_timeout)
        else:
            self._exit_once(DEADLINE_EXIT_CODE, "hard_deadline")

    def _training_timeout(self):
        if self._closed.is_set():
            return
        try:
            remaining = (self.training_stop_at - _aware(self.clock())).total_seconds()
        except Exception as error:
            self.last_error = ErrorInfo("clock", type(error).__name__, str(error))
            self._stop("monitor_error", log=False)
            return
        if remaining > 0:
            self._arm_timer("_training_timer", remaining, self._training_timeout)
        else:
            self._stop("training_deadline")

    def _critical_timeout(self):
        with self._lock:
            if self._critical_due is None or self._stopped.is_set() or self._closed.is_set():
                return
            try:
                remaining = self._critical_due - self.monotonic()
            except Exception as error:
                self.last_error = ErrorInfo("monotonic", type(error).__name__, str(error))
                remaining = 0  # A known critical condition must not gain grace.
        if remaining > 0:
            self._arm_timer("_critical_timer", remaining, self._critical_timeout)
        else:
            self._exit_once(THERMAL_EXIT_CODE, "thermal_critical")

    def _stop_timeout(self):
        with self._lock:
            if (not self._stop_timer_armed or self._stopped.is_set()
                    or self._closed.is_set() or self._hard_exit_requested):
                return
            try:
                remaining = self._stop_due - self.monotonic() if self._stop_due is not None else 0
            except Exception:
                remaining = 0
            reason = self._reason or self._stop_origin_reason or "monitor_error"
        if remaining > 0:
            self._arm_timer("_stop_timer", remaining, self._stop_timeout)
        else:
            # Do not call logger/reader before exiting an unacknowledged stop.
            self._exit_once(STOP_TIMEOUT_EXIT_CODE, reason)

    def _heartbeat_timeout(self):
        with self._lock:
            if self._closed.is_set() or self._stopped.is_set() or self._hard_exit_requested:
                return
            anchor = self._last_success_mono if self._last_success_mono is not None else self._heartbeat_anchor
            if anchor is None:
                return
            try:
                remaining = self.stale_after - (self.monotonic() - anchor)
            except Exception:
                remaining = 0
        if remaining > 0:
            self._arm_timer("_heartbeat_timer", remaining, self._heartbeat_timeout)
        else:
            self.last_error = ErrorInfo("heartbeat", "stale_sample",
                                        f"No complete thermal/power/emit sample for {self.stale_after:g}s")
            self._stop("monitor_error", log=False)

    def check_once(self):
        """Deterministic polling hook for mock tests; normally called by thread."""
        with self._poll_lock:
            if self._closed.is_set() or self._hard_exit_requested:
                return
            try:
                at, mono = _aware(self.clock()), self.monotonic()
                if at >= self.hard_stop_at:
                    self._exit_once(DEADLINE_EXIT_CODE, "hard_deadline")
                    return
                if at >= self.training_stop_at:
                    self._stop("training_deadline")
                if self._stop_timer_armed:
                    self._stop_timeout()
                    if self._hard_exit_requested:
                        return
                if self._critical_due is not None:
                    self._critical_timeout()
                    return  # Stop is latched; no slow reader is needed now.
                thermal = self.read_thermal()
                if not isinstance(thermal, ThermalSample) or not thermal.ok:
                    self.last_error = thermal.error if isinstance(thermal, ThermalSample) else ErrorInfo("thermal", "invalid_sample", repr(thermal))
                    self._stop("monitor_error")
                    return
                self.last_thermal = thermal
                if thermal.state == 3:
                    with self._lock:
                        self._critical_due = mono + self.critical_grace
                    if self._thread is not None:
                        self._arm_timer("_critical_timer", self.critical_grace, self._critical_timeout)
                    self._stop("thermal_critical")
                    return
                if thermal.state == 2:
                    self._stop("thermal_serious")
                if self.monitor_power and (self._next_power is None or mono >= self._next_power):
                    power = self.read_power()
                    if not isinstance(power, PowerSample) or not power.ok:
                        self.last_error = power.error if isinstance(power, PowerSample) else ErrorInfo("power", "invalid_sample", repr(power))
                        self._stop("monitor_error")
                        return
                    self.last_power, self._next_power = power, mono + self.power_interval
                    if power.on_battery and power.percent < 20:
                        self._stop("low_battery")
                emitted = self._log("sample", state=thermal.state, state_name=thermal.name, at=at.isoformat())
                # Readers may consume time; enforce cutoffs again before wait.
                if _aware(self.clock()) >= self.hard_stop_at:
                    self._exit_once(DEADLINE_EXIT_CODE, "hard_deadline")
                elif _aware(self.clock()) >= self.training_stop_at:
                    self._stop("training_deadline")
                if emitted:
                    with self._lock:
                        self._last_success_mono = self.monotonic()
                        self._last_success_at = _aware(self.clock()).isoformat()
                if not self.should_stop:
                    self.ready_event.set()
            except Exception as error:
                self.last_error = ErrorInfo("watchdog", type(error).__name__, str(error))
                self._stop("monitor_error")
            finally:
                self.initial_sample_event.set()

    def _run(self):
        while not self._closed.is_set() and not self._hard_exit_requested:
            self.check_once()
            self._closed.wait(self.sample_interval)

    def start(self):
        with self._lock:
            if self._started or self._closed.is_set():
                raise RuntimeError("NightGuard cannot be started twice or after close")
            self._started = True
        try:
            at = _aware(self.clock())
            self._heartbeat_anchor = self.monotonic()
        except Exception as error:
            self.last_error = ErrorInfo("clock", type(error).__name__, str(error))
            self._stop("monitor_error")
            self.initial_sample_event.set()
            return self
        self._thread = threading.Thread(target=self._run, name="night-thermal-watchdog", daemon=True)
        self._arm_timer("_hard_timer", (self.hard_stop_at - at).total_seconds(), self._hard_timeout)
        self._arm_timer("_training_timer", (self.training_stop_at - at).total_seconds(), self._training_timeout)
        self._arm_timer("_heartbeat_timer", self.stale_after, self._heartbeat_timeout)
        self._thread.start()
        return self

    def mark_stopped(self):
        """All main work/checkpoint I/O ended; call only before close/return."""
        with self._lock:
            self._stopped.set()
            if self._critical_timer is not None:
                self._critical_timer.cancel()
            for timer in (self._heartbeat_timer, self._stop_timer):
                if timer is not None:
                    timer.cancel()

    def close(self, timeout=1.0):
        self._closed.set()
        with self._lock:
            for timer in (self._critical_timer, self._hard_timer, self._training_timer,
                          self._heartbeat_timer, self._stop_timer):
                if timer is not None:
                    timer.cancel()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)

    def metadata(self):
        with self._lock:
            return {"source": "NSProcessInfo.thermalState via Foundation/Objective-C ctypes",
                    "limitation": LIMITATION, "sample_interval_seconds": self.sample_interval,
                    "critical_grace_seconds": self.critical_grace, "power_monitor_enabled": self.monitor_power,
                    "heartbeat_stale_after_seconds": self.stale_after,
                    "heartbeat_success_definition": "Thermal reader, due power reader and sample emit returned; initial freshness anchor is guard start.",
                    "last_complete_sample_at": self._last_success_at,
                    "last_complete_sample_monotonic": self._last_success_mono,
                    "protective_stop_grace_seconds": self.stop_grace,
                    "protective_stop_reasons": ["thermal_serious", "low_battery", "monitor_error"],
                    "protective_stop_deadline_never_extended": True,
                    "protective_stop_origin_reason": self._stop_origin_reason,
                    "protective_stop_deadline_monotonic": self._stop_due,
                    "protective_stop_exit_code": STOP_TIMEOUT_EXIT_CODE,
                    "power_interval_seconds": self.power_interval, "battery_stop_below_percent": 20,
                    "training_stop_at": self.training_stop_at.isoformat(), "hard_stop_at": self.hard_stop_at.isoformat(),
                    "should_stop": self.should_stop, "reason": self._reason, "stop_reasons": list(self._reasons),
                    "initial_ready": self.initial_ready, "all_compute_stopped_acknowledged": self._stopped.is_set(),
                    "hard_exit_requested": self._hard_exit_requested,
                    "last_state": self.last_thermal.name if self.last_thermal is not None else None,
                    "last_thermal_at": self.last_thermal.sampled_at.isoformat() if self.last_thermal is not None else None,
                    "last_power": {"on_battery": self.last_power.on_battery, "percent": self.last_power.percent,
                                   "sampled_at": self.last_power.sampled_at.isoformat()} if self.last_power is not None else None,
                    "last_error": vars(self.last_error) if self.last_error is not None else None,
                    "checkpoint_policy": "No watchdog checkpoint I/O; retain only caller's already-complete atomic checkpoint."}
