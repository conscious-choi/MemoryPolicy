from aloha.real_env import make_real_env
from aloha.robot_utils import move_grippers
from interbotix_common_modules.common_robot.robot import create_interbotix_global_node, get_interbotix_global_node, robot_startup
from interbotix_common_modules.common_robot.exceptions import InterbotixException
from aloha.constants import FOLLOWER_GRIPPER_JOINT_OPEN

def aloha_init_node(args):
    """try to get interbotix node"""
    setup_base = args.aloha.setup_base
    try:
        node = get_interbotix_global_node()
    except:
        node = create_interbotix_global_node('aloha')

    env = make_real_env(node=node, setup_robots=True, setup_base=setup_base)
    
    try:
        robot_startup(node)
    except InterbotixException:
        pass

    return node

def reset_grippers(env):
    move_grippers(
        [env.follower_bot_left, env.follower_bot_right],
        [FOLLOWER_GRIPPER_JOINT_OPEN] * 2,
        moving_time=0.5,
    )  # open