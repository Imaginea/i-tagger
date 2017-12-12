import unittest
import os, sys
from config.config_factory import TFConfigFactory

from config.config_helper import ConfigManager

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.file = "/tmp/config.ini"
        if os.path.exists(self.file):\
            os.remove(self.file)
        self.config: ConfigManager = ConfigManager(self.file)
        self.config.add_section("Test")
        self.config.set_item("Test", "value", "0")

        self.assertEqual(int(self.config.get_item('Test', "value")), 0)

