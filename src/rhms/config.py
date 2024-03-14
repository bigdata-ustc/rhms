# -*- coding:utf-8 -*-

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='Relation-enHanced Math Solver')

    parser.add_argument('--cuda', type=str, dest='cuda_id', default=None)
    parser.add_argument('--mode', type=str, dest='mode',default='train')
    parser.add_argument('--checkpoint', type=str, dest='checkpoint', default="0")
    parser.add_argument('--resume', action='store_true', dest='resume', default=False)
    parser.add_argument('--log', type=str, dest='log', default=None)
    parser.add_argument('--test-log', type=str, dest='test_log', default=None)
    parser.add_argument('--seed', type=int, dest='seed', default=2)

    parser.add_argument('--epoch', type=int, dest='epoch', default=80)
    parser.add_argument('--warm', type=int, dest='warm', default=10)
    parser.add_argument('--batch', type=int, dest='batch', default=64)
    parser.add_argument('--lr', type=float, dest='lr', default=1e-3)
    parser.add_argument('--weight-decay', type=float, dest='weight_decay', default=1e-5)
    parser.add_argument('--step', type=int, dest='step', default=20)
    parser.add_argument('--gamma', type=float, dest='gamma', default=0.5)
    parser.add_argument('--beam', type=int, dest='beam', default=None)

    parser.add_argument('--embed', type=int, dest='embed', default=128)
    parser.add_argument('--bert', type=str, dest='bert', default=None)
    parser.add_argument('--hidden', type=int, dest='hidden', default=512)
    parser.add_argument('--hop', type=int, dest='hop', default=1)
    parser.add_argument('--head', type=int, dest='head', default=4)
    parser.add_argument('--threshold', type=float, dest='threshold', default=0.85)
    parser.add_argument('--dropout', type=float, dest='dropout', default=0.5)

    args = parser.parse_args()
    return args
