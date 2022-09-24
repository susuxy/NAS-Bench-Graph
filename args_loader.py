import argparse

def load_args():
    parser = argparse.ArgumentParser('Train_from_Genotype')
    parser.add_argument('--data',           type = str,             default = 'arxiv')
    args = parser.parse_known_args()[0]
    return args