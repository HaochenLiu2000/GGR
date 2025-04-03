import argparse
import sys

def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_args(parser):
    parser.add_argument('--name', default='webqsp', type=str)
    parser.add_argument('--data_folder', default='data/webqsp/', type=str)
    parser.add_argument('--max_train', default=200000, type=int)

    parser.add_argument('--relation2id', default='relations.txt', type=str)
    parser.add_argument('--entity2id', default='entities.txt', type=str)

    parser.add_argument('--use_self_loop', default=True, type=bool_flag)
    parser.add_argument('--use_inverse_relation', action='store_true')
    parser.add_argument('--data_eff', action='store_true')
    
    parser.add_argument('--llm_name', default='gpt3.5', type=str)

