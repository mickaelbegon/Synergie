import importlib
import sys
import threading
import types
import unittest
from unittest.mock import patch


class FakeDevice:
    def __init__(self, device_id="1", tag="1", recording=False, recording_count=0, plugged=True):
        self.deviceId = device_id
        self.deviceTagName = tag
        self.isRecording = recording
        self.recordingCount = recording_count
        self.isPlugged = plugged
        self.isBatteryCharging = plugged
        self.batteryLevel = 87
        self.timingRecord = 0
        self.currentImage = ""
        self.imageActive = "active"
        self.imageInactive = "inactive"
        self.export_args = None

    def exportData(self, save_file, event):
        self.export_args = (save_file, isinstance(event, threading.Event))
        event.set()


class FakeDotManager:
    def getExportEstimatedTime(self):
        return 1


class FakeCheckbutton:
    def __init__(self, selected=False):
        self.selected = selected

    def instate(self, states):
        return self.selected and states == ["selected"]


class ImmediateThread:
    def __init__(self, target, args=(), daemon=False):
        self.target = target
        self.args = args
        self.daemon = daemon

    def start(self):
        self.target(*self.args)


def _fake_ttkbootstrap_module():
    module = types.ModuleType("ttkbootstrap")

    class Frame:
        pass

    class Window:
        pass

    class Toplevel:
        pass

    class Style:
        def configure(self, *args, **kwargs):
            return None

    module.Frame = Frame
    module.Window = Window
    module.Toplevel = Toplevel
    module.Style = Style
    return module


def _import_front_module(module_name):
    dot_device = types.ModuleType("core.utils.DotDevice")
    dot_device.DotDevice = FakeDevice

    database_manager = types.ModuleType("core.database.DatabaseManager")
    database_manager.DatabaseManager = object
    database_manager.TrainingData = object

    dot_manager = types.ModuleType("core.utils.DotManager")
    dot_manager.DotManager = FakeDotManager

    with patch.dict(
        sys.modules,
        {
            "core.utils.DotDevice": dot_device,
            "core.database.DatabaseManager": database_manager,
            "core.utils.DotManager": dot_manager,
            "ttkbootstrap": _fake_ttkbootstrap_module(),
        },
    ):
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)


class DotInterfaceUiTests(unittest.TestCase):
    def test_dot_frame_status_tracks_sensor_state(self):
        module = _import_front_module("front.DotFrame")
        frame = module.DotFrame.__new__(module.DotFrame)
        frame.device = FakeDevice(recording_count=0)

        self.assertEqual(frame._status(), ("Prêt", "SuccessBadge.TLabel"))

        frame.device.recordingCount = 2
        self.assertEqual(frame._status(), ("À exporter", "WarningBadge.TLabel"))

        frame.device.isRecording = True
        self.assertEqual(frame._status(), ("Enregistrement", "SuccessBadge.TLabel"))

        frame.device.isRecording = False
        frame.device.recordingCount = 0
        frame.device.isPlugged = False
        self.assertEqual(frame._status(), ("Débranché", "NeutralBadge.TLabel"))

    def test_dot_frame_image_follows_usb_state(self):
        module = _import_front_module("front.DotFrame")
        frame = module.DotFrame.__new__(module.DotFrame)
        frame.device = FakeDevice(plugged=True)

        self.assertEqual(frame._display_image(), "active")

        frame.device.isPlugged = False
        self.assertEqual(frame._display_image(), "inactive")

    def test_dot_frame_recording_message_mentions_pending_export(self):
        module = _import_front_module("front.DotFrame")
        frame = module.DotFrame.__new__(module.DotFrame)
        frame.device = FakeDevice(recording=False, recording_count=3, plugged=True)

        self.assertIn("Données présentes", frame._recording_message())
        self.assertIn("exportez", frame._recording_message())

    def test_main_page_export_all_sends_boolean_save_option(self):
        module = _import_front_module("front.MainPage")
        device = FakeDevice(recording=False, recording_count=1, plugged=True)
        page = module.MainPage.__new__(module.MainPage)
        page.dotsConnected = [device]
        page.estimatedTime = 1
        page.saveFile = FakeCheckbutton(selected=False)

        with patch.object(module.threading, "Thread", ImmediateThread), patch.object(module, "ExtractingPage"):
            page.export_all_dots()

        self.assertEqual(device.export_args, (False, True))

    def test_main_page_export_all_skips_recording_or_empty_devices(self):
        module = _import_front_module("front.MainPage")
        recording = FakeDevice(device_id="recording", recording=True, recording_count=1, plugged=True)
        empty = FakeDevice(device_id="empty", recording=False, recording_count=0, plugged=True)
        unplugged = FakeDevice(device_id="unplugged", recording=False, recording_count=1, plugged=False)
        page = module.MainPage.__new__(module.MainPage)
        page.dotsConnected = [recording, empty, unplugged]
        page.estimatedTime = 1
        page.saveFile = FakeCheckbutton(selected=True)

        with patch.object(module.threading, "Thread", ImmediateThread), patch.object(module, "ExtractingPage"):
            page.export_all_dots()

        self.assertIsNone(recording.export_args)
        self.assertIsNone(empty.export_args)
        self.assertIsNone(unplugged.export_args)


if __name__ == "__main__":
    unittest.main()
