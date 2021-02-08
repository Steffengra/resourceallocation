from shutil import copy2
from policygradient_config import Config


config = Config()

copy2('policygradient_config.py', config.model_path)