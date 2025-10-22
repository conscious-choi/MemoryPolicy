def general_parser(parser):
    parser.add_argument(
        "--exp_name",
        default="memory_policy",
        type=str
    )
    parser.add_argument(
        '--inference',
        action='store_true',
        help='Evaluate the selected model checkpoint',
    )
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        help='Checkpoint directory',
    )
    parser.add_argument(
        '--policy_class',
        type=str,
        default='MemoryPolicy',
        help='The desired policy class',
    )
    parser.add_argument(
        '--task_name',
        type=str,
        help='Name of the task. Must be in task configurations',
    )
    parser.add_argument(
        '--state_dim',
        type=int,
        default=14,
        help='state dim of robot',
    )
    parser.add_argument(
        '--action_dim',
        type=int,
        default=16,
        help='state dim of robot',
    )


    parser.add_argument(
        "--use_wandb",
        action="store_true"
    )
    parser.add_argument(
        "--wandb.project_name",
        default="MemoryPolicy",
        type=str
    )
    parser.add_argument(
        "--wandb.entity",
        default="s-choi-org",
        type=str
    )

    return parser