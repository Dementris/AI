import networkx as nx
import random


class Car():
    def __init__(self, graph: nx.Graph, start, finish):
        self.graph = graph
        self.current_node = start
        self.direction = None
        self.finish = finish
        self.visited_neighbors = set(self.current_node)
        self.path = []
        self.direction_update()
        self.current_actions = []
        self.full_path = []

    def get_neighbors(self):
        return list(self.graph.neighbors(self.current_node))

    def is_finish(self):
        return self.current_node == self.finish

    def action(self):
        neighbors = self.get_neighbors()
        if self.is_there_road(neighbors):
            self.current_node = self.direction
            self.direction_update()
            return "action"
        else:return "no_way"


    def direction_update(self):
        self.direction = (self.current_node[0], self.current_node[1] + 1)

    def direction_to_left(self):
        self.direction = (self.current_node[0] - 1, self.current_node[1])

    def direction_to_right(self):
        self.direction = (self.current_node[0] + 1, self.current_node[1])

    def direction_to_180(self):
        self.direction = (self.current_node[0], self.current_node[1] - 1)

    def is_there_road(self, neighbors):
        return self.direction in neighbors

    def set_direction(self,next_node):
        if next_node == (self.current_node[0] - 1, self.current_node[1]):
            self.direction_to_left()
        elif next_node == (self.current_node[0] + 1, self.current_node[1]):
            self.direction_to_right()
        elif next_node == (self.current_node[0], self.current_node[1] - 1):
            self.direction_to_180()
        else:self.direction_update()

    def choose_direction_and_action(self):
        neighbors = self.get_neighbors()
        unvisited_neighbors = [n for n in neighbors if n not in self.visited_neighbors]
        if unvisited_neighbors:
            next_node = random.choice(unvisited_neighbors)
            self.set_direction(next_node)
            self.visited_neighbors.add(self.current_node)
            self.path.append(self.current_node)
            self.full_path.append(self.current_node)
        else:
            if not self.path:
                return
            next_node = self.path.pop()
            self.set_direction(next_node)

        self.action()
        self.visited_neighbors.add(self.current_node)

    def navigate(self):
        while not self.is_finish():
            self.choose_direction_and_action()
        self.path.append(self.finish)
        self.full_path.append(self.finish)
        for i in range(len(self.path)-1):
            self.graph[self.path[i]][self.path[i+1]]['color'] = 'red'
        for i in range(len(self.full_path)-1):
            if self.full_path[i] == self.full_path[i+1]:
                self.graph.add_edge(self.full_path[i],self.full_path[i+1])
            self.graph[self.full_path[i]][self.full_path[i+1]]['color'] = 'green'


