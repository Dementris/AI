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
        self.direction_edges_list = []
        self.knowledge_base = {"routes": {},
                               "visited_nodes": set(self.current_node)}

    def ask(self,current_node_information):
        """
        Ask knowledge base for information.
        """
        return self.knowledge_base["routes"][current_node_information], self.knowledge_base["visited_nodes"]


    def tell(self):
        """
        Add information to knowledge base.
        """
        neighbors = self.get_neighbors()
        self.knowledge_base["routes"][self.current_node] = neighbors
        self.knowledge_base["visited_nodes"].add(self.current_node)


    def get_neighbors(self) -> list:
        """
        Neighbors of current node.
        :return list
        """
        return list(self.graph.neighbors(self.current_node))

    def is_finish(self) -> bool:
        """
        Is finish.
        :return: bool
        """
        return self.current_node == self.finish

    def action(self):
        """
        Move forward to destination.
        :return: str
        """
        neighbors = self.get_neighbors()
        if self.is_there_road(neighbors):
            self.current_node = self.direction
            self.direction_update()
            self.full_path.append(self.current_node)
            return "action"
        else:return "no_way"

    def direction_update(self):
        """
        Update direction to current node.
        """
        self.direction = (self.current_node[0], self.current_node[1] + 1)

    def direction_to_left(self):
        """
        Direction to left.
        """
        self.direction = (self.current_node[0] - 1, self.current_node[1])

    def direction_to_right(self):
        """
        Direction to right.
        """
        self.direction = (self.current_node[0] + 1, self.current_node[1])

    def direction_to_180(self):
        """
        A 180 degree turn.
        """
        self.direction = (self.current_node[0], self.current_node[1] - 1)

    def is_there_road(self, neighbors: list) -> bool:
        """
        Check is there road.
        :param neighbors: list
        :return: bool
        """
        return self.direction in neighbors

    def set_direction(self,next_node):
        """
        Set direction to next node
        :param next_node:
        """
        if next_node == (self.current_node[0] - 1, self.current_node[1]):
            self.direction_to_left()
        elif next_node == (self.current_node[0] + 1, self.current_node[1]):
            self.direction_to_right()
        elif next_node == (self.current_node[0], self.current_node[1] - 1):
            self.direction_to_180()
        else:self.direction_update()

    def choose_direction_and_action(self):
        """
        Method that randomly selects the path to the next neighbouring node.
        Generates a map of the travelled path.
        """
        neighbors = self.get_neighbors()
        unvisited_neighbors = [n for n in neighbors if n not in self.visited_neighbors]
        if unvisited_neighbors:
            next_node = random.choice(unvisited_neighbors)
            self.set_direction(next_node)
            self.visited_neighbors.add(self.current_node)
            self.path.append(self.current_node)
        else:
            if not self.path:
                return
            next_node = self.path.pop()
            self.set_direction(next_node)

        self.action()
        self.visited_neighbors.add(self.current_node)

    def navigate(self):
        """
        Method which navigate our car to finish.
        """
        self.full_path.append(self.current_node)
        while not self.is_finish():
            self.choose_direction_and_action()
        for i in range(len(self.full_path)-1):
            self.direction_edges_list.append((self.full_path[i],self.full_path[i+1]))