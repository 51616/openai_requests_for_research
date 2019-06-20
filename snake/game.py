import numpy as np
import random
from PIL import Image
import torch
import torch.nn.functional as F
import cv2

RIGHT_MOVE = 0
DOWN_MOVE = 1
LEFT_MOVE = 2
UP_MOVE = 3

FOOD = 1
BODY = -1

FOOD_REWARD = 1
MOVE_REWARD = 0
DEATH_PENALTY = -1


class Snake():
    def __init__(self, board_size=20):
        self.action_space = [0, 1, 2, 3]
        self.board_size = board_size

    def create_food(self, y=None, x=None):
        if x is None or y is None:
            empty_pos = list(zip(np.where(self.board_body == 0)[
                             0], np.where(self.board_body == 0)[1]))
            food_pos = random.sample(empty_pos, 1)[0]
        else:
            food_pos = (y, x)
        self.board_food[food_pos[0], food_pos[1]] = 1
        return food_pos

    def reset(self):
        self.board_body = np.zeros((self.board_size, self.board_size), dtype='int')
        self.board_head = np.zeros((self.board_size, self.board_size), dtype='int')
        self.board_food = np.zeros((self.board_size, self.board_size), dtype='int')
        self.board_tail = np.zeros((self.board_size, self.board_size), dtype='int')
        

        rand_y = np.random.randint(self.board_size)  # (self.board_size+1)//2
        self.body = [(rand_y,i) for i in range(5)] #self.board_size//2+1
        for i, (y, x) in enumerate(self.body):
            self.board_body[y, x] = i+1
        self.head = (self.body[-1][0], self.body[-1][1])
        self.tail = (self.body[0][0], self.body[0][1])
        self.board_head[self.head[0], self.head[1]] = 1
        self.board_tail[self.tail[0], self.tail[1]] = 1
        self.facing = RIGHT_MOVE
        self.food = self.create_food()
        return (self.board_body.copy(),self.board_head.copy(),
        self.board_tail.copy(),self.board_food.copy(), np.ones((self.board_size,self.board_size))*len(self.body))

    def step(self, action):  # return obs,reward,done
        # check if input is valid action
        if action not in self.action_space:
            return ((self.board_body.copy(),self.board_head.copy(),
        self.board_tail.copy(),self.board_food.copy(), np.ones((self.board_size,self.board_size))*len(self.body)), -1, True)

        # refuse to go backward
        if abs(action-self.facing) == 2:
            action = self.facing

        # print('Current pos:', self.body[-1][0], self.body[-1][1])
        if action == RIGHT_MOVE:
            next_y, next_x = (self.body[-1][0], self.body[-1][1] + 1)
        elif action == DOWN_MOVE:
            next_y, next_x = (self.body[-1][0] + 1, self.body[-1][1])
        elif action == LEFT_MOVE:
            next_y, next_x = (self.body[-1][0], self.body[-1][1] - 1)
        else:
            next_y, next_x = (self.body[-1][0] - 1, self.body[-1][1])

        # print('Next pos:', next_y, next_x)

        self.facing = action  # remember last direction

        # print('Current board')
        # print(self.board)

        # hit the wall
        if next_x == self.board_size or next_y == self.board_size or next_x < 0 or next_y < 0:
            return ((self.board_body.copy(),self.board_head.copy(),
        self.board_tail.copy(),self.board_food.copy(), np.ones((self.board_size,self.board_size))*len(self.body)), DEATH_PENALTY, True)
        # hit body
        if self.board_body[next_y, next_x]:
            return ((self.board_body.copy(),self.board_head.copy(),
        self.board_tail.copy(),self.board_food.copy(), np.ones((self.board_size,self.board_size))*len(self.body)), DEATH_PENALTY, True)
        # hit the food
        if self.board_food[next_y, next_x]:

            self.board_head[next_y, next_x] = 1
            
            self.body.append((next_y, next_x))
            self.board_body[next_y, next_x] = len(self.body)
            # self.board_body[self.head[0], self.head[1]] = 1
            self.board_head[self.head[0], self.head[1]] = 0
            self.board_food[next_y, next_x] = 0
            self.head = (next_y, next_x)

            if len(self.body)==self.board_size**2:
                return ((self.board_body.copy(),self.board_head.copy(),
                            self.board_tail.copy(),self.board_food.copy(),
                            np.ones((self.board_size,self.board_size))*len(self.body)),
                            FOOD_REWARD, True)
            
            self.food = self.create_food()

            return ((self.board_body.copy(),self.board_head.copy(),
        self.board_tail.copy(),self.board_food.copy(), np.ones((self.board_size,self.board_size))*len(self.body)), FOOD_REWARD, False)

        # normal move
        else:
            self.board_body = F.relu(torch.from_numpy(self.board_body - 1)).numpy()
            self.board_head[next_y, next_x] = 1
            self.board_body[next_y, next_x] = len(self.body)
            self.board_body[self.body[0][0], self.body[0][1]] = 0
            self.board_tail[self.tail[0], self.tail[1]] = 0

            self.body.append((next_y, next_x))            
            self.body = self.body[1:]
           
            self.tail = (self.body[0][0], self.body[0][1])
            self.board_tail[self.tail[0], self.tail[1]] = 1

            # self.board_body[self.head[0], self.head[1]] = 1
            self.board_head[self.head[0], self.head[1]] = 0
            self.head = (next_y, next_x)

            # print('New board')
            # print(self.board)

            return ((self.board_body.copy(),self.board_head.copy(),
        self.board_tail.copy(),self.board_food.copy(), np.ones((self.board_size,self.board_size))*len(self.body)), MOVE_REWARD, False)

    def render(self):
        render_arr = np.zeros(
            (self.board_size, self.board_size, 3), dtype=np.uint8)

        snake_pos = self.body
        for y, x in snake_pos:
            render_arr[y, x] = np.array((128, 128, 0))

        head = self.head
        render_arr[head[0], head[1]] = np.array((255, 128, 0))

        food_pos = self.food
        render_arr[food_pos[0], food_pos[1]] = np.array((0, 255, 0))

        img = Image.fromarray(render_arr, 'RGB')
        img = img.resize((300, 300))
        cv2.imshow('', np.array(img))
        cv2.waitKey(100)
