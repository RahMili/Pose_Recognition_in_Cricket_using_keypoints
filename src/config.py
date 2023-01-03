import configparser
import os

CONFIG_DIR = 'src/config/'
CONFIG_FILE_NAME = "config.ini"
basic_config = configparser.ConfigParser()
basic_config.read(os.path.join(CONFIG_DIR, CONFIG_FILE_NAME))