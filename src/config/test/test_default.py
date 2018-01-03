# import unittest
#
# from helpers.config_helper import *
# from config.default import create_default_config
#
# class TestDefault(unittest.TestCase):
#
#     def test_creation(self):
#         config_helper = ConfigManager("/tmp/config.ini")
#         if not os.path.exists("/tmp/config.ini"):
#             create_default_config(config_helper)
#         self.assertEqual(os.path.exists("/tmp/config.ini"), True)
#         self.assertEqual(config_helper.get_item("Schema", "text_column"), "word")
#
#
