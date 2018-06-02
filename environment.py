import numpy  as np
import time


class happy(object):

    def __init__(self, H, W, max_num, seed):
        self.H = H
        self.W = W
        self.max_num = max_num
        self.points = 0
        self.reset(seed)


    def reset(self, seed):
        np.random.seed(seed)
        self.sample_pool = np.empty((0, self.H, self.W), dtype='uint8')
        for i in range(1, self.max_num+1):
            weight = self.max_num-i+1
            self.sample_pool = np.concatenate((self.sample_pool, np.ones((weight, self.H, self.W))*i), axis=0)
        self.sample_pool = self.sample_pool.reshape(-1)
        self.map = np.random.choice(self.sample_pool, self.W*self.H).reshape(self.W, self.H)
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

                # if find the target
                if len(index_list) >= 3:
                    print 'target node: ', (i, j), ' target number: ', self.map[i, j], 'target area: ', len(index_list)
                    column_dict = self.singleupdate(index_list)
                    self.is_visited = np.zeros((self.H, self.W), dtype=bool)
                    print self.map


    def singleupdate(self, index_list, command_index=None):
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
            target = command_index
        else:
            left_column = index_list[0][1]
            left_list = column_dict[left_column]
            target = left_list[0]

        column_dict[target[1]].remove(target)

        # update
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


    def feedempty(self, column_dict):
        for c in column_dict:
            for node in column_dict[c]:
                self.map[node] = np.random.choice(self.sample_pool)




if __name__ == '__main__':
    env = happy(5, 5, 9, 1234)
    env.allsearch()
