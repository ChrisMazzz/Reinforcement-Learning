class Schedul_env():
    action = 0
    r = 0

    def __init__(self):
        self.viewer = None
        self.times = [10, 8, 15, 5, 5, 20, 10, 10, 3, 12, 15, 3, 8, 10, 12]
        self.human = 54  # 实际上计算55个未分配的人就好
        self.maxcloths = 300
        self.start_state = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.real_state = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.step_count = 0
        self.counts = 0
        self.output_time = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.min_state = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.actions = 15
        self.min = 99999
        self.done = 0

    def time(self):
        t = 0
        C = [a / b for a, b in zip(self.times, self.real_state)]
        for i in range(self.actions - 1):
            t += max(C[0:i + 1]) + max(C[i:self.actions])
        t += (self.maxcloths - self.actions + 1) * max(C)
        return t

    def cherk_reward(self, s, done):
        if done == 0:
            reward = self.time()
            reward += -99999
            realtime = reward
            # print(' ')
            # print(self.real_state)
            # print(realtime)
            return reward
        else:
            reward = self.time()
            # for i in range(14):
            #     self.output_time[i] = self.time[i] / s[i]
            #     sum += self.output_time[i]
            realtime = reward
            print(' ')
            print(self.real_state)
            print(realtime)
            return reward

    def step(self, actions):
        a = actions
        if self.counts == 0:
            self.real_state = self.start_state
            self.real_state[a] += 1
            self.counts += 1
            done = self.done
            r = self.cherk_reward(self.real_state, done)
            return self.real_state, r, done

        elif self.counts < self.human:
            self.real_state[a] += 1
            self.counts += 1
            done = 0
            r = self.cherk_reward(self.real_state, done)
            return self.real_state, r, done

        else:
            self.real_state[a] += 1
            done = 1
            r = self.cherk_reward(self.real_state, done)
            self.start_state = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            self.real_state = self.start_state
            self.counts = 0
            x = done
            if r < self.min:
                self.min = r
                self.min_state = self.real_state
            print('当前最佳时间为：')
            print(self.min)
            print('当前最佳情况为：')
            print(self.min_state)
            self.done = 0
            return self.real_state, r, x



