from .architectures import architecture_parser
from .training import training_parser
from .inference import inference_parser
from .general import general_parser
from .dataset import dataset_parser

def arg_parser(parser):
    parser = architecture_parser(parser)
    parser = training_parser(parser)
    parser = inference_parser(parser)
    parser = general_parser(parser)
    parser = dataset_parser(parser)
    return parser