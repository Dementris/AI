import networkx as nx
import random
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from num_recognition import NumCNN

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
        self.speed = []
        self.model = NumCNN()
        self.model.load_state_dict(torch.load('model.pth'))
        self.model.eval()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

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
        :return: list
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
        neighbors, _ = self.ask(self.current_node)
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
        self.tell()
        neighbors, visited_nodes = self.ask(self.current_node)
        unvisited_neighbors = [n for n in neighbors if n not in visited_nodes]
        if unvisited_neighbors:
            next_node = random.choice(unvisited_neighbors)
            self.set_direction(next_node)
            self.path.append(self.current_node)
        else:
            if not self.path:
                return
            next_node = self.path.pop()
            self.set_direction(next_node)
        # Speed limit sign recognition
        image_x1, image_y1 = self.node_to_image_position(self.direction[0],self.direction[1])
        image_x2, image_y2 = self.node_to_image_position(self.current_node[0], self.current_node[1])
        sign = self.extract_edge_image(image_x1, image_y1, image_x2, image_y2)
        sign = self.transform(sign)
        with torch.no_grad():
            output = self.model(sign.unsqueeze(0))
            _, predicted = torch.max(output, 1)
            predicted_class = predicted.item()
        print(f"Predicted speed for edge {self.current_node},{self.direction} = {predicted_class}0 km/h")
        self.speed.append({f"Edge ({self.current_node},{self.direction})": f"Speed:{predicted_class}0 km/h"})
        self.action()

    def node_to_image_position(self, node_x, node_y):
        # Image position of the starting node (0, 0)
        start_image_x, start_image_y = 56.0, 439.0

        # Horizontal and vertical distance between nodes on the image
        horizontal_distance = 132
        vertical_distance = 100

        image_x = start_image_x + node_x * horizontal_distance
        image_y = start_image_y - node_y * vertical_distance
        return image_x, image_y

    def extract_edge_image(self, node_center_x1, node_center_y1, node_center_x2, node_center_y2, node_size=39, image_path='road_map.png'):
        road_map = cv2.imread(image_path)

        top_left_x = int(node_center_x1 - node_size / 2)
        top_left_y = int(node_center_y1 - node_size / 2)
        bottom_right_x = int(node_center_x1 + node_size / 2)
        bottom_right_y = int(node_center_y1 + node_size / 2)

        top_left_x1 = int(node_center_x2 - node_size / 2)
        top_left_y1 = int(node_center_y2 - node_size / 2)
        bottom_right_x1 = int(node_center_x2 + node_size / 2)
        bottom_right_y1 = int(node_center_y2 + node_size / 2)

        edge_image = road_map[min(top_left_y, top_left_y1):max(bottom_right_y, bottom_right_y1),
                     min(top_left_x, top_left_x1):max(bottom_right_x, bottom_right_x1)]

        new_width, new_height = 28, 28

        height, width = edge_image.shape[:2]

        center_x, center_y = width // 2, height // 2

        start_x = max(center_x - (new_width // 2), 0)
        end_x = min(center_x + (new_width // 2), width)
        start_y = max(center_y - (new_height // 2), 0)
        end_y = min(center_y + (new_height // 2), height)

        cropped_image = edge_image[start_y:end_y, start_x:end_x]

        resized_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        inverted_image = 255 - resized_image
        img_gray = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)

        return img_gray

    def navigate(self):
        """
        Method which navigate our car to finish.
        """
        self.full_path.append(self.current_node)
        while not self.is_finish():
            self.choose_direction_and_action()
        for i in range(len(self.full_path)-1):
            self.direction_edges_list.append((self.full_path[i],self.full_path[i+1]))