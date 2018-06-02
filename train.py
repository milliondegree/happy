from environment import happy
from RL_brain import *
import argparse


def get_argument():
	parser = argparse.ArgumentParser()
	parser.add_argument('--episode', default=1000)
	parser.add_argument('--height', default=5)
	parser.add_argument('--width', default=5)
	parser.add_argument('--max_num', default=5)		
	parser.add_argument('--seed', default=1234)


def train()
	args = get_argument()
	env = happy(args.height, args.width, args.max_num, args.seed)
	action_space = []
	for i in args.height:
		for j in args.width:
			axtion_space.append((i, j))
	 
