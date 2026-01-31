import yaml
from utils.utils import Struct

def load_config(path):
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Struct(**config_dict)
