import numpy as np

class MetaSampler:
    def __init__(self, n_sources=15, strategy='group_fixed_order', groups=None):
        self.n_sources = n_sources
        self.strategy = strategy
        self.groups = groups if groups else [[i] for i in range(n_sources)]
        self.reset()

    def reset(self):
        """重置：只打乱每组内部顺序，组间顺序保持原始定义顺序"""
        if self.strategy == 'fixed':
            self.order = list(range(self.n_sources))
            self.iter = iter(self.order)
        elif self.strategy == 'random':
            self.order = np.random.permutation(self.n_sources).tolist()
            self.iter = iter(self.order)
        elif self.strategy == 'group_fixed_order':
            # 组间顺序：按 groups 列表的顺序执行（不打乱）
            self.group_queues = []
            for group in self.groups:
                # 只打乱组内顺序
                shuffled_group = np.random.permutation(group).tolist()
                self.group_queues.append(shuffled_group)
            # 当前处理到第几个组
            self.current_group_index = 0
        elif self.strategy == 'group_sequential':
            # 1. 打乱组的顺序
            self.group_order = np.random.permutation(len(self.groups)).tolist()
            # 2. 为每个组生成随机排列的队列
            self.group_queues = []
            for group_idx in self.group_order:
                group_data = self.groups[group_idx]
                shuffled_group = np.random.permutation(group_data).tolist()
                self.group_queues.append(shuffled_group)
            # 3. 当前处理到第几个组
            self.current_queue_index = 0
            self.current_queue = None
            # 4. 初始化第一个组的队列
            if self.group_queues:
                self.current_queue = self.group_queues[0]

    def __next__(self):
        if self.strategy in ['fixed', 'random']:
            try:
                return next(self.iter)
            except StopIteration:
                self.reset()
                return next(self.iter)

        elif self.strategy == 'group_fixed_order':
            # 遍历所有组，直到当前组有数据
            while self.current_group_index < len(self.group_queues):
                current_queue = self.group_queues[self.current_group_index]
                if current_queue:
                    return current_queue.pop(0)  # 返回组内随机打乱后的下一个
                else:
                    # 当前组空了，进入下一组
                    self.current_group_index += 1

            # 所有组都遍历完了，重置（重新打乱组内顺序）
            self.reset()
            return self.__next__()
        elif self.strategy == 'group_sequential':
            # 如果当前组还有数据，返回一个
            if self.current_queue and self.current_queue_index < len(self.group_queues):
                if self.current_queue:
                    return self.current_queue.pop(0)

            # 当前组空了，尝试切换到下一组
            self.current_queue_index += 1
            if self.current_queue_index < len(self.group_queues):
                self.current_queue = self.group_queues[self.current_queue_index]
                return self.current_queue.pop(0)

            # 所有组都遍历完了，触发重置
            self.reset()
            return self.__next__()

    def __iter__(self):
        return self

if __name__ == '__main__':
    sampler = MetaSampler(
        n_sources=15,
        strategy='fixed',
        groups=[[0,1,2,3], [4,5,6,7,8,9], [10,11,12,13], [14]]
    )

    for _ in range(32):
        idx = next(sampler)
        # data = next(meta_iters[idx])
        print(idx)