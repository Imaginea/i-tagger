import os
import configparser
from configparser import ExtendedInterpolation
from helpers.print_helper import *

class ConfigManager(object):
    def __init__(self, config_path: str):
        # set the path to the config file
        self.config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
        self.config_path = config_path

        if os.path.exists(config_path):
            self.config.read(self.config_path)
        else:
            self.save_config()

    def set_item(self, section: str, option: str, value: str):
        self.config.set(section=section,
                        option=option,
                        value=value)

    def get_item(self, section, option)-> str:
        return self.config.get(section=section,
                               option=option)
    def add_section(self, section):
        self.config.add_section(section)

    def get_item_as_int(self,section, option):
        return self.config.getint(section=section,
                               option=option)

    def get_item_as_boolean(self,section, option):
        return self.config.getboolean(section=section,
                               option=option)

    def save_config(self):

        print_info(os.getcwd())
        # Writing our configuration file to 'config'
        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)