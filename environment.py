import numpy  as np
import time


class happy(object):

    def __init__(self, H, W, max_num, seed = None):
        if seed:
            np.random.seed(seed)
        self.H = H
        self.W = W
        self.max_num = max_num
        self.sample_pool = np.empty((0, self.H, self.W), dtype='uint8')
        for i in range(1, self.max_num+1):
            weight = self.max_num-i+1
            self.sample_pool = np.concatenate((self.sample_pool, np.ones((weight, self.H, self.W))*i), axis=0)
        self.sample_pool = self.sample_pool.reshape(-1)
        self.action_space = []
        for i in range(H):
            for j in range(W):
                self.action_space.append((i, j))
        self.reset()


    def reset(self):
        self.steps = 0
        self.points = 0
        self.life = 5
        self.map = np.random.choice(self.sample_pool, self.W*self.H).reshape(self.W, self.H)
        _ = self.allsearch(if_init=True)
        print 'initialization succeeded! '
        print self.map, '\n'
        return np.concatenate([self.map.reshape(-1), np.array([self.life])], axis=0)


    def dsp(self, i, j, num, index_list):
        if i<0 or j<0 or i>=self.H or j>=self.W or self.is_visited[i, j]:
            return
        else:
            if self.map[i, j] == num:
                index_list.append((i, j))
                self.is_visited[i, j] = True
                self.dsp(i+1, j, num, index_list)
                self.dsp(i, j+1, num, index_list)
                self.dsp(i-1, j, num, index_list)
                self.dsp(i, j-1, num, index_list)


    def singlesearch(self, index):
        self.steps += 1
        self.map[index] += 1
        self.life -= 1
        print 'pressing at ', index
        print self.map, 'life:', self.life, 'points:', self.points, '\n'
        self.is_visited = np.zeros((self.H, self.W), dtype=bool)
        index_list = []
        self.dsp(index[0], index[1], self.map[index], index_list)
        if len(index_list) >= 3:
            self.points += self.map[index] * len(index_list) * 10
            if self.life < 5:
                self.life += 1
            self.singleupdate(index_list, command_index=index)
            print self.map, 'happy at', index, ' life:', self.life, ' points:', self.points, '\n'
            combo = self.allsearch()+1
        else:
            print 'life:', self.life, 'points:', self.points
            combo = 0
        print 'combo:', combo
        return combo


    def allsearch(self, if_init=False):
        self.is_visited = np.zeros((self.H, self.W), dtype=bool)
        for i in range(self.H):
            for j in range(self.W):
                if self.is_visited[i, j]:
                    continue
                index_list = []
                self.dsp(i, j, self.map[i, j], index_list)

                # if find the target
                if len(index_list) >= 3:
                    if not if_init:
                        # arranging points and lifes
                        self.points += self.map[i, j] * len(index_list) * 10
                        if self.life < 5:
                            self.life += 1
                        index = self.singleupdate(index_list)
                        print self.map, 'happy at ', index, 'life:', self.life, 'points:', self.points, '\n'
                        return self.allsearch()+1
                    # when initializing, we do not need to add the increase the value
                    else:
                        _ = self.singleupdate(index_list, if_add=False)
                        return self.allsearch(if_init=True)+1
        return 0


    def singleupdate(self, index_list, command_index=None, if_add=True):
        # sort the index_list and make the column dictionary
        index_list = sorted(index_list, key=lambda ele:ele[1])
        columns = set(zip(*index_list)[1])
        column_dict = {}
        for c in columns:
            column_dict[c] = []
        for node in index_list:
            column_dict[node[1]].append(node)

        # sort every column list
        for c in columns:
            column_dict[c] = sorted(column_dict[c], key=lambda ele:ele[0], reverse=True)

        # find the target node
        if command_index:
            target_column = command_index[1]
            target = column_dict[target_column][0]
        else:
            left_column = index_list[0][1]
            left_list = column_dict[left_column]
            target = left_list[0]


        # update
        if if_add:
            column_dict[target[1]].remove(target)
            self.map[target] += 1
        for c in columns:
            if len(column_dict[c]) == 0:
                continue
            deepest = column_dict[c][0][0]
            for i in reversed(range(deepest+1)):
                if i-len(column_dict[c]) < 0:
                    self.map[i, c] = np.random.choice(self.sample_pool)
                else:
                    new_node = (i-len(column_dict[c]), c)
                    self.map[i, c] = self.map[new_node]
        return target


    def step(self, action):
        pre_points = self.points
        combo = self.singlesearch(self.action_space[action])
        reward = self.points - pre_points
        s_ = np.concatenate([self.map.reshape(-1), np.array([self.life])], axis=0)
        if self.life == 0:
            done = True
            reward = -100000+self.points
        else:
            done = False
        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        print self.map, '\n'


if __name__ == '__main__':
    env = happy(5, 5, 5, 1234)
    while env.life > 0:
        s = raw_input('input index:')
        l = list(s)
        i = int(l[0])
        j = int(l[-1])
        if i>=0 and i<env.H and j>=0 and j<env.W:
            combo = env.singlesearch((i, j))
        else:
            print 'input error!'
            break
    print 'you die'
