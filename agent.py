import networkx as nx
import random


class Car():
    def __init__(self, graph: nx.Graph, start, finish):
        self.graph = graph
        self.current_node = start
        self.direction = None
        self.finish = finish
        self.visited_map = set(self.current_node)
        self.path = []
        self.direction_update()
        self.current_actions = []

    def get_neighbors(self):
        return list(self.graph.neighbors(self.current_node))

    def is_finish(self):
        return self.current_node == self.finish

    def action(self):
        neighbors = self.get_neighbors()
        if self.is_there_road(neighbors):
            self.update_color()
            self.current_node = self.direction
            self.direction_update()
            return "action"
        else:return "stop"

    def update_color(self):
        self.graph[self.current_node][self.direction]['color'] = 'red'

    def direction_update(self):
        self.direction = (self.current_node[0], self.current_node[1] + 1)

    def direction_to_left(self):
        self.direction = (self.current_node[0] - 1, self.current_node[1])

    def direction_to_right(self):
        self.direction = (self.current_node[0] + 1, self.current_node[1])


    def is_there_road(self, neighbors):
        return self.direction in neighbors

    def chose_next_action(self):
        pass


    def navigate(self):
        while not self.is_finish():
            pass

