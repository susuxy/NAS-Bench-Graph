import argparse

def load_args():
    parser = argparse.ArgumentParser('Nas-Bench-Graph')
    parser.add_argument('--data',           type = str,             default = 'arxiv')
    parser.add_argument('--encode', choices=['one_hot', 'category'], default='one_hot', help='encoding method for category variables')
    args = parser.parse_known_args()[0]
    return args