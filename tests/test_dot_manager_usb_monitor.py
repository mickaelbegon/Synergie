import importlib
import sys
import types
import unittest
from unittest.mock import patch


class FakeDevice:
    def __init__(self, device_id, charging=True, recording=False, recording_count=0):
        self.deviceId = device_id
        self.isBatteryCharging = charging
        self.isPlugged = True
        self.isRecording = recording
        self.recordingCount = recording_count
        self.closed = 0
        self.opened = 0

    def closeUsb(self):
        self.closed += 1
        self.isPlugged = False
        self.isBatteryCharging = False

    def openUsb(self):
        self.opened += 1
        self.isPlugged = True
        self.isBatteryCharging = True
        return True


def _import_dot_manager_with_fakes():
    dot_device = types.ModuleType("core.utils.DotDevice")
    dot_device.DotDevice = FakeDevice

    database_manager = types.ModuleType("core.database.DatabaseManager")
    database_manager.DatabaseManager = object

    xdpchandler = types.ModuleType("core.utils.xdpchandler")
    xdpchandler.XdpcHandler = object
    xdpchandler.XsPortInfo = object
    xdpchandler.XsDotConnectionManager = object
    xdpchandler.XsDotDevice = object
    xdpchandler.windows_com_ports = lambda: []

    numpy = types.ModuleType("numpy")
    numpy.max = max

    with patch.dict(
        sys.modules,
        {
            "core.utils.DotDevice": dot_device,
            "core.database.DatabaseManager": database_manager,
            "core.utils.xdpchandler": xdpchandler,
            "numpy": numpy,
        },
    ):
        sys.modules.pop("core.utils.DotManager", None)
        return importlib.import_module("core.utils.DotManager")


class DotManagerUsbMonitorTests(unittest.TestCase):
    def setUp(self):
        module = _import_dot_manager_with_fakes()
        self.manager = module.DotManager(db_manager=object())
        self.manager.usbTransitionThreshold = 2

    def test_unplugging_one_sensor_only_reports_that_sensor_after_debounce(self):
        sensor_1 = FakeDevice("1")
        sensor_2 = FakeDevice("2")
        sensor_10 = FakeDevice("10")
        self.manager.devices = [sensor_1, sensor_2, sensor_10]
        self.manager.previousConnected = [sensor_1, sensor_2, sensor_10]
        self.manager.usbMonitorPrimed = True

        sensor_1.isBatteryCharging = False

        first_connected, first_disconnected = self.manager.checkDevices()
        second_connected, second_disconnected = self.manager.checkDevices()

        self.assertEqual(first_connected, [])
        self.assertEqual(first_disconnected, [])
        self.assertEqual(second_connected, [])
        self.assertEqual(second_disconnected, [sensor_1])
        self.assertEqual(sensor_1.closed, 1)
        self.assertEqual(sensor_2.closed, 0)
        self.assertEqual(sensor_10.closed, 0)

    def test_plugging_back_one_sensor_reports_only_that_sensor_after_debounce(self):
        sensor_1 = FakeDevice("1", charging=False)
        sensor_2 = FakeDevice("2")
        sensor_1.isPlugged = False
        self.manager.devices = [sensor_1, sensor_2]
        self.manager.previousConnected = [sensor_2]
        self.manager.usbMonitorPrimed = True

        sensor_1.isBatteryCharging = True

        first_connected, first_disconnected = self.manager.checkDevices()
        second_connected, second_disconnected = self.manager.checkDevices()

        self.assertEqual(first_connected, [])
        self.assertEqual(first_disconnected, [])
        self.assertEqual(second_connected, [sensor_1])
        self.assertEqual(second_disconnected, [])
        self.assertEqual(sensor_1.opened, 1)
        self.assertEqual(sensor_2.opened, 0)

    def test_simultaneous_unplug_and_replug_reports_both_transitions(self):
        unplugged = FakeDevice("1")
        stable = FakeDevice("2")
        replugged = FakeDevice("10", charging=False)
        replugged.isPlugged = False
        self.manager.devices = [unplugged, stable, replugged]
        self.manager.previousConnected = [unplugged, stable]
        self.manager.usbMonitorPrimed = True

        unplugged.isBatteryCharging = False
        replugged.isBatteryCharging = True

        self.manager.checkDevices()
        connected, disconnected = self.manager.checkDevices()

        self.assertEqual(connected, [replugged])
        self.assertEqual(disconnected, [unplugged])
        self.assertEqual(unplugged.closed, 1)
        self.assertEqual(replugged.opened, 1)
        self.assertEqual(stable.closed, 0)
        self.assertEqual(stable.opened, 0)


if __name__ == "__main__":
    unittest.main()
