def inference_parser(parser):
    parser.add_argument(
        '--infer.temporal_agg',
        action='store_true',
    )
    parser.add_argument(
        '--infer.ckpt_name',
        type='str',
        required=True
    )
    parser.add_argument(
        '--infer.onscreen_render',
        action='store_true'
    )
    parser.add_argument(
        "--infer.num_rollouts",
        type=int,
        default=1
    )

    return parser