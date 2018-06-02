import numpy  as np
import time


class happy(object):

    def __init__(self, H, W, max_num, seed):
        self.H = H
        self.W = W
        self.max_num = max_num
        self.reset(seed)


    def reset(self, seed):
        np.random.seed(seed)
        self.sample_pool = np.empty((0, self.H, self.W), dtype='uint8')
        for i in range(1, self.max_num+1):
            weight = self.max_num-i+1
            self.sample_pool = np.concatenate((self.sample_pool, np.ones((weight, self.H, self.W))*i), axis=0)
        self.map = np.random.choice(self.sample_pool.reshape(-1), self.W*self.H).reshape(self.W, self.H)
        self.is_visited = np.zeros((self.H, self.W), dtype=bool)
        print 'initialization succeeded! '
        print self.map, '\n'


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


    def allsearch(self):
        self.is_visited = np.zeros((self.H, self.W), dtype=bool)
        for i in range(self.H):
            for j in range(self.W):
                if self.is_visited[i, j]:
                    continue
                index_list = []
                self.dsp(i, j, self.map[i, j], index_list)
                print self.map[i, j], index_list


    def singleupdate(self, index_list, command_index=None):
        # sort the index_list and make the column dictionary
        index_list = sorted(index_list, key=lambda ele:ele[1])
        columns = set(zip(*index_list)[1])
        column_dict = {}
        for c in columns:
            column_dict[c] = []
        for node in index_list:
            column_dict[node[1]].append(node)

        # find the node the left and bottom
        if command_index:
            target = command_index
        else:
            left_column = index_list[0][1]
            left_list = column_dict[left_column]
            left_list = sorted(left_list, key=lambda ele:ele[0], reverse=True)
            target = left_list[0]
        column[]
        # update
        self.map[target] += 1



if __name__ == '__main__':
    env = happy(4, 4, 5, 1234)
    env.allsearch()
