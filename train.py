from environment import happy
from DeepQNet import DeepQNetwork
from plot import *
import argparse
import logging
import os
import numpy as np


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=1000)
    parser.add_argument('--height', type=int, default=5)
    parser.add_argument('--width', type=int, default=5)
    parser.add_argument('--max_num', type=int, default=5)
    parser.add_argument('--episode_num', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--reward_decay', type=float, default=0.9)
    parser.add_argument('--e_greedy', type=float, default=0.9)
    parser.add_argument('--model_name', type=str)
    return parser.parse_args()


def dqn_train():
    args = get_argument()
    logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename='./log/'+args.model_name+'.log',
        filemode='w'
        )
    logging.info('random seed: %d' % args.seed)

    env = happy(args.height, args.width, args.max_num, args.seed)
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
    x_list = []
    y_list = []
    for e in range(args.episode_num):
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
                if e % 100 == 0:
                    x_list.append(e)
                    y_list.append(env.points)
                if env.points > max_points:
                    x_list.append(e)
                    y_list.append(env.points)
                    print 'game %d done, points: %d' % (e, env.points)
                    logging.info('game %d finished, points: %d' % (e, env.points))
                    max_points = env.points
                    dqn.saver.save(dqn.sess, './save/'+args.model_name+'_best')
                break

    if not os.path.exists('./plot/npz'):
        os.makedirs('./plot/npz')
    np.savez('./plot/npz/'+args.model_name+'.npz', x=np.array(x_list), y=np.array(y_list))
    print 'training over'
    print max_points



def random_game():
    # random game
    args = get_argument()
    x_list = []
    y_list = []
    max_points = 0
    env = happy(args.height, args.width, args.max_num, args.seed)
    for e in range(100000):
        env.reset()
        while True:
            action = np.random.randint(0, args.height*args.width)
            obz_, reward, done = env.step(action)
            if done:
                if e % 10 == 0:
                    x_list.append(e)
                    y_list.append(env.points)
                if env.points > max_points:
                    print 'game %d done, points: %d' % (e, env.points)
                    max_points = env.points
                break
    if not os.path.exists(Base_Path):
        os.makedirs(Base_Path)
    np.savez(Base_Path+'/'+args.model_name+'.npz', x=np.array(x_list), y=np.array(y_list))
    print 'random game over'
    print max_points
    drawlineplot(args.model_name+'.npz')




if __name__ == '__main__':
    # random_game()
    dqn_train()
