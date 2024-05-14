import json

def get_configuration(path_to_conf):
    """
    Load a configuration file from the specified path.

    Parameters:
    - path_to_conf (str): Path to the configuration file.

    Returns:
    - dict: The loaded configuration as a dictionary.
    """
    print("[CONF] Loading configuration file {}".format(path_to_conf))

    conf = json.load(open(path_to_conf, "r"))

    return conf
