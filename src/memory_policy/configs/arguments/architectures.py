def architecture_parser(parser):
    # High Level policy

    # Low Level Policy
    parser.add_argument(
        '--decoder.chunk_size',
        type=int,
        help='chunk_size',
    )
    parser.add_argument(
        '--decoder.hidden_dim',
        type=int,
        help='hidden_dim',
    )
    parser.add_argument(
        '--decoder.dim_feedforward',
        type=int,
        help='dim_feedforward',
    )
    parser.add_argument(
        '--decoder.num_decoder_layes',
        type=int,
        help='num_decoder_layes',
        default=3
    )
    parser.add_argument(
        '--decoder.num_heads',
        type=int,
        help='num_heads',
        default=8
    )

    # For WM Module
    

    return parser