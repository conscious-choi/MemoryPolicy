def dataset_parser(parser):
    parser.add_argument(
        '--task.dataset_dir_list',
        type=str,
        help='The desired policy class',
    )
    return parser