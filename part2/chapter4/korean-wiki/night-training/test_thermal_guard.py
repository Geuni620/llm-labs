"""Mock-only tests: no Foundation/pmset reads, heat load or real os._exit."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

from thermal_guard import (DEADLINE_EXIT_CODE, STOP_TIMEOUT_EXIT_CODE, THERMAL_EXIT_CODE, ErrorInfo,
                           MacOSThermalReader, NightGuard, PowerSample, ThermalSample,
                           parse_power_sample, read_power_sample)


KST = timezone(timedelta(hours=9))
START = datetime(2026, 10, 22, 23, 40, tzinfo=KST)
STOP = datetime(2026, 10, 23, 6, 30, tzinfo=KST)
HARD = datetime(2026, 10, 23, 7, 0, tzinfo=KST)


class FakeClock:
    def __init__(self):
        self.wall, self.mono = START, 0.0

    def now(self):
        return self.wall

    def advance(self, seconds):
        self.wall += timedelta(seconds=seconds)
        self.mono += seconds


class FakeTimer:
    def __init__(self, delay, callback):
        self.delay, self.callback = delay, callback
        self.daemon, self.started, self.canceled = False, False, False

    def start(self):
        self.started = True

    def cancel(self):
        self.canceled = True

    def fire(self):
        self.callback()


class ThermalGuardTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.guards = []
        self.timers = []
        self.exits = []
        # A bug or omitted injection must still never reach a real OS exit/read.
        self.exit_patch = patch("thermal_guard.os._exit", side_effect=AssertionError("Real exit forbidden"))
        self.foundation_patch = patch("thermal_guard.ctypes.CDLL", side_effect=AssertionError("Live Foundation read forbidden"))
        self.process_patch = patch("thermal_guard.subprocess.run", side_effect=AssertionError("Live process query forbidden"))
        for item in (self.exit_patch, self.foundation_patch, self.process_patch):
            item.start()

    def tearDown(self):
        for guard in self.guards:
            guard.close()
        for item in (self.process_patch, self.foundation_patch, self.exit_patch):
            item.stop()

    def timer(self, delay, callback):
        timer = FakeTimer(delay, callback)
        self.timers.append(timer)
        return timer

    def thermal(self, state=0):
        return ThermalSample(self.clock.now(), state, True)

    def power(self, percent=80, battery=False):
        return PowerSample(self.clock.now(), battery, percent)

    def guard(self, **overrides):
        values = dict(clock=self.clock.now, monotonic=lambda: self.clock.mono,
                      read_thermal=lambda: self.thermal(), read_power=lambda: self.power(),
                      hard_exit=self.exits.append, timer_factory=self.timer)
        values.update(overrides)
        result = NightGuard(STOP, HARD, **values)
        self.guards.append(result)
        return result

    def test_reader_mock_states_and_lazy_single_load(self):
        getter = Mock(side_effect=[0, 1, 2, 3])
        loader = Mock(return_value=getter)
        reader = MacOSThermalReader(clock=self.clock.now, platform="darwin", loader=loader)
        self.assertEqual(loader.call_count, 0)
        self.assertEqual([reader.sample_now().name for _ in range(4)], ["nominal", "fair", "serious", "critical"])
        loader.assert_called_once()

    def test_reader_failure_unsupported_and_invalid_are_explicit(self):
        loader = Mock(side_effect=NotImplementedError("missing selector"))
        sample = MacOSThermalReader(clock=self.clock.now, platform="darwin", loader=loader).sample_now()
        self.assertFalse(sample.ok)
        self.assertTrue(sample.error.unsupported)
        other = MacOSThermalReader(clock=self.clock.now, platform="linux", loader=Mock()).sample_now()
        self.assertFalse(other.supported)
        invalid = MacOSThermalReader(clock=self.clock.now, platform="darwin", loader=lambda: lambda: 9).sample_now()
        self.assertEqual(invalid.error.kind, "invalid_state")

    def test_ready_requires_successful_initial_thermal_and_power(self):
        guard = self.guard()
        self.assertFalse(guard.initial_ready)
        guard.check_once()
        self.assertTrue(guard.wait_ready(0))
        self.assertFalse(guard.should_stop)
        self.assertIn("unsupported hardware", guard.metadata()["limitation"])

    def test_initial_power_failure_prevents_ready(self):
        guard = self.guard(read_power=lambda: PowerSample(self.clock.now(), None, None, ErrorInfo("power", "error", "mock")))
        guard.check_once()
        self.assertTrue(guard.initial_sample_event.is_set())
        self.assertFalse(guard.wait_ready(0))
        self.assertEqual(guard.reason, "monitor_error")

    def test_fair_continues_serious_latches_even_after_cooling(self):
        states = iter([1, 2, 0])
        guard = self.guard(read_thermal=lambda: self.thermal(next(states)))
        guard.check_once()
        self.assertFalse(guard.should_stop)
        guard.check_once()
        self.assertTrue(guard.should_stop)
        guard.check_once()
        self.assertEqual(guard.reason, "thermal_serious")
        self.assertFalse(guard.initial_ready)

    def test_critical_stop_then_exit_by_five_monotonic_seconds(self):
        guard = self.guard(read_thermal=lambda: self.thermal(3))
        guard.check_once()
        self.assertEqual(guard.reason, "thermal_critical")
        self.clock.advance(4.999)
        guard.check_once()
        self.assertEqual(self.exits, [])
        self.clock.advance(.001)
        guard.check_once()
        guard.check_once()
        self.assertEqual(self.exits, [THERMAL_EXIT_CODE])

    def test_acknowledged_all_compute_stopped_cancels_critical_exit(self):
        guard = self.guard(read_thermal=lambda: self.thermal(3))
        guard.check_once()
        guard.mark_stopped()
        self.clock.advance(10)
        guard.check_once()
        self.assertEqual(self.exits, [])
        self.assertTrue(guard.should_stop)

    def test_absolute_deadlines_not_relative_to_late_start(self):
        self.clock.wall = STOP - timedelta(seconds=1)
        guard = self.guard()
        guard.check_once()
        self.clock.advance(1)
        guard.check_once()
        self.assertEqual(guard.reason, "training_deadline")
        guard.mark_stopped()
        self.clock.wall = HARD
        guard.check_once()
        self.assertEqual(self.exits, [DEADLINE_EXIT_CODE])

    def test_wall_clock_rollback_does_not_extend_critical_grace(self):
        guard = self.guard(read_thermal=lambda: self.thermal(3))
        guard.check_once()
        self.clock.wall -= timedelta(hours=2)
        self.clock.mono = 5
        guard.check_once()
        self.assertEqual(self.exits, [THERMAL_EXIT_CODE])

    def test_reader_exception_bad_sample_and_logger_error_stop(self):
        for reader in (Mock(side_effect=RuntimeError("mock read error")), lambda: object()):
            guard = self.guard(read_thermal=reader)
            guard.check_once()
            self.assertEqual(guard.reason, "monitor_error")
            self.assertFalse(guard.wait_ready(0))
        guard = self.guard(emit=Mock(side_effect=RuntimeError("mock log error")))
        guard.check_once()
        self.assertEqual(guard.reason, "monitor_error")

    def test_battery_threshold_and_30_second_sampling(self):
        reader = Mock(side_effect=[self.power(20, True), self.power(19, True)])
        guard = self.guard(read_power=reader)
        guard.check_once()
        self.assertFalse(guard.should_stop)
        self.clock.advance(29)
        guard.check_once()
        self.assertEqual(reader.call_count, 1)
        self.clock.advance(1)
        guard.check_once()
        self.assertEqual(guard.reason, "low_battery")
        ac = self.guard(read_power=lambda: self.power(1, False))
        ac.check_once()
        self.assertFalse(ac.should_stop)

    def test_optional_power_disable_never_calls_reader(self):
        reader = Mock(side_effect=AssertionError("disabled reader called"))
        guard = self.guard(monitor_power=False, read_power=reader)
        guard.check_once()
        self.assertTrue(guard.initial_ready)
        reader.assert_not_called()

    def test_pmset_parser_and_mocked_subprocess(self):
        battery = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t19%; discharging; 1:00 remaining\n"
        ac = "Now drawing from 'AC Power'\n"
        self.assertEqual(parse_power_sample(battery, sampled_at=self.clock.now()).percent, 19)
        self.assertTrue(parse_power_sample(ac, sampled_at=self.clock.now()).ok)
        self.assertFalse(parse_power_sample("unrecognized", sampled_at=self.clock.now()).ok)
        run = Mock(return_value=SimpleNamespace(returncode=0, stdout=battery, stderr=""))
        self.assertTrue(read_power_sample(clock=self.clock.now, run=run).on_battery)
        self.assertEqual(run.call_args.kwargs["timeout"], 2)
        self.assertFalse(read_power_sample(clock=self.clock.now, run=Mock(side_effect=TimeoutError("mock"))).ok)

    def test_thread_starts_with_mock_inputs_and_closes(self):
        guard = self.guard().start()
        self.assertTrue(guard.wait_ready(.5))
        self.assertTrue(self.timers[0].started)
        guard.close()
        self.assertFalse(guard._thread.is_alive())
        self.assertTrue(all(timer.canceled for timer in self.timers))
        with self.assertRaises(RuntimeError):
            guard.start()

    def test_deadline_timer_survives_blocked_reader(self):
        entered, release = threading.Event(), threading.Event()
        def blocked():
            entered.set()
            release.wait(1)
            return self.thermal()
        guard = self.guard(read_thermal=blocked).start()
        try:
            self.assertTrue(entered.wait(.5))
            self.clock.wall = HARD
            self.timers[0].fire()
            self.assertEqual(self.exits, [DEADLINE_EXIT_CODE])
        finally:
            release.set()
            guard.close()

    def test_critical_timer_survives_blocked_logger(self):
        entered, release = threading.Event(), threading.Event()
        def blocked_log(record):
            entered.set()
            release.wait(1)
        guard = self.guard(read_thermal=lambda: self.thermal(3), emit=blocked_log).start()
        try:
            self.assertTrue(entered.wait(.5))
            self.assertEqual(len(self.timers), 4)
            self.clock.mono = 5
            guard._critical_timer.fire()
            self.assertEqual(self.exits, [THERMAL_EXIT_CODE])
        finally:
            release.set()
            guard.close()

    def test_close_cancels_every_hard_exit(self):
        guard = self.guard().start()
        self.assertTrue(guard.wait_ready(.5))
        guard.mark_stopped()
        guard.close()
        self.clock.wall = HARD
        for timer in self.timers:
            timer.fire()
        guard.check_once()
        self.assertEqual(self.exits, [])

    def test_training_timer_stops_at_fixed_time_even_with_blocked_reader(self):
        entered, release = threading.Event(), threading.Event()
        def blocked():
            entered.set()
            release.wait(1)
            return self.thermal()
        guard = self.guard(read_thermal=blocked).start()
        try:
            self.assertTrue(entered.wait(.5))
            self.clock.wall = STOP
            self.timers[1].fire()
            self.assertEqual(guard.reason, "training_deadline")
            self.assertEqual(self.exits, [])
        finally:
            release.set()
            guard.close()

    def test_initial_clock_error_is_not_ready_and_cannot_restart(self):
        guard = self.guard(clock=Mock(side_effect=RuntimeError("mock clock failure")))
        guard.start()
        self.assertFalse(guard.wait_ready(0))
        self.assertEqual(guard.reason, "monitor_error")
        self.assertIsNone(guard._heartbeat_timer)
        self.assertTrue(guard._stop_timer.started)
        self.assertEqual(guard._stop_timer.delay, 30)
        with self.assertRaises(RuntimeError):
            guard.start()

    def test_monitor_error_and_critical_upgrade_deadline_reason(self):
        self.clock.wall = STOP
        reader = Mock(side_effect=[self.thermal(2), self.thermal(3)])
        guard = self.guard(read_thermal=reader)
        guard.check_once()
        self.assertEqual(guard.reason, "thermal_serious")
        guard.check_once()
        self.assertEqual(guard.reason, "thermal_critical")
        self.assertEqual(guard.metadata()["stop_reasons"], ["training_deadline", "thermal_serious", "thermal_critical"])

    def test_power_exception_and_pmset_exit_failure_are_errors(self):
        guard = self.guard(read_power=Mock(side_effect=TimeoutError("mock power timeout")))
        guard.check_once()
        self.assertEqual(guard.reason, "monitor_error")
        self.assertFalse(guard.initial_ready)
        run = Mock(return_value=SimpleNamespace(returncode=1, stdout="", stderr="mock failure"))
        result = read_power_sample(clock=self.clock.now, run=run)
        self.assertFalse(result.ok)
        self.assertIn("exited 1", result.error.message)

    def test_known_critical_never_extends_grace_on_monotonic_error(self):
        guard = self.guard(read_thermal=lambda: self.thermal(3))
        guard.check_once()
        guard.monotonic = Mock(side_effect=RuntimeError("mock monotonic failure"))
        guard._critical_timeout()
        self.assertEqual(self.exits, [THERMAL_EXIT_CODE])

    def test_initial_sample_crossing_stop_time_is_never_ready(self):
        def delayed_power():
            self.clock.wall = STOP
            return self.power()
        guard = self.guard(read_power=delayed_power)
        guard.check_once()
        self.assertFalse(guard.wait_ready(0))
        self.assertFalse(guard.ready_event.is_set())
        self.assertEqual(guard.reason, "training_deadline")

    def _blocked_after_ready(self, component):
        entered, release = threading.Event(), threading.Event()
        guard = self.guard().start()
        self.assertTrue(guard.wait_ready(.5))
        self.assertEqual(guard.metadata()["last_complete_sample_monotonic"], 0)
        def blocked_reader():
            entered.set()
            release.wait(1)
            return self.thermal()
        def blocked_logger(record):
            entered.set()
            release.wait(1)
        if component == "reader":
            guard.read_thermal = blocked_reader
        else:
            guard.emit = blocked_logger
        self.clock.advance(5)
        poll = threading.Thread(target=guard.check_once)
        poll.start()
        try:
            self.assertTrue(entered.wait(.5))
            self.clock.advance(15)
            guard._heartbeat_timer.fire()
            self.assertTrue(guard.should_stop)
            self.assertEqual(guard.reason, "monitor_error")
            self.assertEqual(guard.last_error.kind, "stale_sample")
            self.assertEqual(guard.metadata()["last_complete_sample_monotonic"], 0)
            self.assertEqual(guard._stop_due, 50)
            self.clock.advance(29)
            guard._stop_timeout()
            self.assertEqual(self.exits, [])
            self.clock.advance(1)
            guard._stop_timer.fire()
            self.assertEqual(self.exits, [STOP_TIMEOUT_EXIT_CODE])
        finally:
            release.set()
            poll.join(.5)
            guard.close()
            self.assertFalse(poll.is_alive())

    def test_stale_reader_after_ready_requests_stop_and_forces_ack_deadline(self):
        self._blocked_after_ready("reader")

    def test_stale_logger_after_ready_requests_stop_and_forces_ack_deadline(self):
        self._blocked_after_ready("logger")

    def test_serious_stop_timer_arms_before_blocked_stop_logger(self):
        entered, release = threading.Event(), threading.Event()
        def blocked_logger(record):
            entered.set()
            release.wait(1)
        guard = self.guard(read_thermal=lambda: self.thermal(2), emit=blocked_logger).start()
        try:
            self.assertTrue(entered.wait(.5))
            self.assertEqual(guard.reason, "thermal_serious")
            self.assertEqual(guard._stop_due, 30)
            self.clock.advance(20)
            guard._heartbeat_timer.fire()
            self.assertEqual(guard.reason, "monitor_error")
            self.assertEqual(guard._stop_due, 30)
            self.clock.advance(10)
            guard._stop_timer.fire()
            self.assertEqual(self.exits, [STOP_TIMEOUT_EXIT_CODE])
        finally:
            release.set()
            guard.close()

    def test_successful_sample_refreshes_independent_heartbeat(self):
        guard = self.guard().start()
        self.assertTrue(guard.wait_ready(.5))
        self.clock.advance(15)
        guard.check_once()
        self.clock.advance(5)
        guard._heartbeat_timer.fire()
        self.assertFalse(guard.should_stop)
        self.assertEqual(guard._heartbeat_timer.delay, 15)
        self.clock.advance(15)
        guard._heartbeat_timer.fire()
        self.assertEqual(guard.reason, "monitor_error")
        self.assertEqual(guard._stop_due, 65)

    def test_serious_low_battery_and_monitor_errors_force_after_30_seconds(self):
        readers = [dict(read_thermal=lambda: self.thermal(2)),
                   dict(read_power=lambda: self.power(19, True)),
                   dict(read_thermal=Mock(side_effect=RuntimeError("mock read error")))]
        for values in readers:
            self.exits.clear()
            self.clock = FakeClock()
            guard = self.guard(**values)
            guard.check_once()
            self.assertTrue(guard.should_stop)
            self.assertEqual(guard._stop_due, 30)
            self.clock.advance(29.9)
            guard._stop_timeout()
            self.assertEqual(self.exits, [])
            self.clock.advance(.1)
            guard._stop_timer.fire()
            self.assertEqual(self.exits, [STOP_TIMEOUT_EXIT_CODE])
            guard.close()

    def test_repeated_stops_cannot_extend_acknowledgment_grace(self):
        guard = self.guard(read_thermal=lambda: self.thermal(2))
        guard.check_once()
        self.clock.advance(29)
        guard.check_once()
        self.assertEqual(guard._stop_due, 30)
        self.clock.advance(1)
        guard._stop_timer.fire()
        self.assertEqual(self.exits, [STOP_TIMEOUT_EXIT_CODE])

    def test_mark_stopped_cancels_stale_and_cooperative_exit(self):
        guard = self.guard(read_thermal=lambda: self.thermal(2)).start()
        self.assertTrue(guard.initial_sample_event.wait(.5))
        self.assertEqual(guard.reason, "thermal_serious")
        guard.mark_stopped()
        self.assertTrue(guard._stop_timer.canceled)
        self.assertTrue(guard._heartbeat_timer.canceled)
        self.clock.advance(60)
        guard._stop_timer.fire()
        guard._heartbeat_timer.fire()
        self.assertEqual(self.exits, [])

    def test_later_critical_keeps_its_earlier_five_second_exit(self):
        values = iter([2, 3])
        guard = self.guard(read_thermal=lambda: self.thermal(next(values))).start()
        self.assertTrue(guard.initial_sample_event.wait(.5))
        self.clock.advance(10)
        guard.check_once()
        self.assertEqual(guard._critical_due, 15)
        self.assertEqual(guard._stop_due, 30)
        self.clock.advance(5)
        guard._critical_timer.fire()
        self.assertEqual(self.exits, [THERMAL_EXIT_CODE])

    def test_stale_monitor_grace_uses_monotonic_despite_wall_rollback(self):
        guard = self.guard().start()
        self.assertTrue(guard.wait_ready(.5))
        self.clock.advance(20)
        guard._heartbeat_timer.fire()
        self.clock.wall -= timedelta(hours=2)
        self.clock.mono = 50
        guard._stop_timer.fire()
        self.assertEqual(self.exits, [STOP_TIMEOUT_EXIT_CODE])

    def test_invalid_deadlines_and_slow_sampling_rejected(self):
        with self.assertRaises(ValueError):
            NightGuard(STOP.replace(tzinfo=None), HARD)
        with self.assertRaises(ValueError):
            NightGuard(HARD, STOP)
        with self.assertRaises(ValueError):
            NightGuard(STOP, HARD, sample_interval=6)
        with self.assertRaises(ValueError):
            NightGuard(STOP, HARD, stale_after=31)
        with self.assertRaises(ValueError):
            NightGuard(STOP, HARD, stop_grace=31)


if __name__ == "__main__":
    unittest.main()
