import sys, os
from jsonargparse import ArgumentParser

from memory_policy.configs.constants import ROOT
from memory_policy.configs.arguments import arg_parser

def initialize(env_parser):
    parser = ArgumentParser(description="MemoryPolicy")
    parser = arg_parser(parser) # common parser
    parser = env_parser(parser) # env parser

    args = parser.parse_args()
    return args