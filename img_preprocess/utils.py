import time
import random
import sys
import yaml


def time_string():
    """Convert datetime to string"""
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string


def convert_secs2time(epoch_time):
    """Convert epoch time"""
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def time_file_str():
    """Create filename with datetime"""
    ISOTIMEFORMAT = '%Y-%m-%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string + '-{}'.format(random.randint(1, 10000))


def print_log(print_string, log):
    """Print Log"""
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()


def load_config(project_path):
    """Load config.yaml file"""
    try:
        with open(project_path + 'config.yaml') as file:
            cfg = yaml.safe_load(file)
            print(cfg)
    except Exception as e:
        print('Exception occurred while loading YAML...', file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    return cfg
