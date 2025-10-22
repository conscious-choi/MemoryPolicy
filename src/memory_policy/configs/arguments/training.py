def training_parser(parser):
    parser.add_argument(
        '--train.seed',
        type=int,
        help='Training seed',
    )
    parser.add_argument(
        '--train.batch_size',
        type=int,
        help='Training batch size',
    )
    parser.add_argument(
        '--train.num_steps',
        type=int,
        help='Number of training steps',
    )
    parser.add_argument(
        '--train.lr',
        type=float,
        help='Training learning rate',
    )
    parser.add_argument(
        '--train.validate_every',
        type=int,
        default=500,
        help='Number of steps between validations during training',
    )
    parser.add_argument(
        '--train.save_every',
        type=int,
        default=500,
        help='Number of steps between checkpoints during training',
    )
    parser.add_argument(
        '--train.resume_ckpt_path',
        type=str,
        help='Path to checkpoint to resume training from',
    )
    parser.add_argument(
        '--train.kl_weight',
        type=int,
        help='KL Weight',
    )

    return parser