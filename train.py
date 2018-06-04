from environment import happy
from DeepQNet import DeepQNetwork
import argparse


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=1000)
    parser.add_argument('--height', type=int, default=5)
    parser.add_argument('--width', type=int, default=5)
    parser.add_argument('--max_num', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--reward_decay', type=float, default=0.9)
    parser.add_argument('--e_greedy', type=float, default=0.9)
    return parser.parse_args()


def dqn_train():
    args = get_argument()
    env = happy(args.height, args.width, args.max_num)
    dqn = DeepQNetwork(25, 26,
            learning_rate=args.learning_rate,
            reward_decay=args.reward_decay,
            e_greedy=args.e_greedy,
            replace_target_iter=200,
            memory_size=2000
            )
    # start training
    steps = 0
    max_points = 0
    for e in range(100000):
        obz = env.reset()
        while True:
            action = dqn.choose_action(obz)
            obz_, reward, done = env.step(action)
            dqn.store_transition(obz, action, reward, obz_)
            if steps>200 and steps%5==0:
                dqn.learn()
            obz = obz_
            steps += 1
            if done:
                if env.points > max_points:
                    max_points = env.points
                break

    print 'training over'
    print max_points


if __name__ == '__main__':
    dqn_train()
