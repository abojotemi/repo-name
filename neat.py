import math
import random
from typing import Callable


class Node:
    def __init__(self, node_id: int, node_type: str):
        self.id = node_id
        self.type = node_type
        self.value = 0.0

    def activate(self, input_sum: float) -> float:
        self.value = 1 / (1 + math.exp(input_sum))
        return self.value


class Connection:
    def __init__(
        self,
        in_node: Node,
        out_node: Node,
        weight: float,
        enabled: bool = True,
        innovation_number: int = 0,
    ):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation_number = innovation_number


class Genome:
    def __init__(self):
        self.nodes: dict[int, Node] = {}
        self.connections: list[Connection] = []
        self.fitness = 0.0

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_connection(self, connection: Connection) -> None:
        self.connections.append(connection)

    def compute_node_values(self, node_id: int, computed: dict[int, float]) -> float:
        if node_id in computed:
            return computed[node_id]
        node = self.nodes[node_id]
        if node.type == "input":
            computed[node_id] = node.value
            return node.value
        total_input = 0.0

        for conn in self.connections:
            if conn.enabled and conn.out_node.id == node.id:
                total_input += conn.weight * self.compute_node_values(
                    conn.in_node.id, computed
                )
        value = node.activate(total_input)
        computed[node_id] = value
        return value

    def forward(self, input_values: list[float]) -> list[float]:
        computed = {}
        input_nodes = sorted(
            [node for node in self.nodes.values() if node.type == "input"],
            key=lambda n: n.id,
        )
        for i, node in enumerate(input_nodes):
            node.value = input_values[i]
            computed[node.id] = node.value

        output_nodes = sorted(
            [node for node in self.nodes.values() if node.type == "output"], key = lambda n: n.id
        )
        outputs = []
        for node in output_nodes:
            outputs.append(self.compute_node_values(node.id, computed))
        return outputs
    
    
class Population:
    def __init__(self, population_size: int, num_inputs: int, num_outputs: int) -> None:
        self.population_size = population_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.genomes: list[Genome] = []
        self.generation: int = 0
        self._create_initial_population()
        
    def _create_initial_population(self) -> None:
        for _ in range(self.population_size):
            genome = self._create_initial_genome()
            self.genomes.append(genome)
    
    def _create_initial_genome(self) -> Genome:
        genome = Genome()
        for i in range(self.num_inputs):
            node = Node(i, "input")
            genome.add_node(node)
        
        for j in range(self.num_outputs):
            node = Node(self.num_inputs + j, 'output')
            genome.add_node(node)
        
        for i in range(self.num_inputs):
            for j in range(self.num_outputs):
                in_node = genome.nodes[i]
                out_node = genome.nodes[self.num_inputs + j]
                weight = random.uniform(-1,1)
                connection = Connection(in_node=in_node, out_node=out_node, weight=weight, enabled=True, innovation_number=0)
                genome.add_connection(connection)
        return genome

    def evaluate(self, fitness_function: Callable[[Genome], float]) -> None:
        for genome in self.genomes:
            genome.fitness = fitness_function(genome)
    
    def select_parents(self) -> tuple[Genome,Genome]:
        tournament_size = 3

        candidates = random.sample(self.genomes, tournament_size)
        parent1 = max(candidates, key=lambda g: g.fitness)
        
        candidates = random.sample(self.genomes, tournament_size)
        parent2 = max(candidates, key=lambda g: g.fitness)
        
        return parent1, parent2

        
    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        child = Genome()
        
        for node_id, node in parent1.nodes.items():
            child_node = Node(node_id, node.type)
            child.add_node(child_node)
        
        conn_dict1 = {(conn.in_node.id, conn.out_node.id): conn for conn in parent1.connections}
        conn_dict2 = {(conn.in_node.id, conn.out_node.id): conn for conn in parent2.connections}

        all_keys = set(conn_dict1.keys()).union(conn_dict2.keys())
        
        for key in all_keys:
            if key in conn_dict1 and key in conn_dict2:
                chosen_conn = random.choice([conn_dict1[key], conn_dict2[key]])
            elif key in conn_dict1:
                chosen_conn = conn_dict1[key]
            else:
                chosen_conn = conn_dict2[key]
            
            in_node = child.nodes[chosen_conn.in_node.id]
            out_node = child.nodes[chosen_conn.out_node.id]
            new_conn = Connection(in_node, out_node, chosen_conn.weight, enabled=chosen_conn.enabled, innovation_number=chosen_conn.innovation_number)
            child.add_connection(new_conn)
        return child
    
    def mutate(self, genome: Genome, weight_perturbation: float = 0.1, add_connection_prob: float = 0.0, add_node_prob: float = 0.03) -> None:
        for conn in genome.connections:
            if random.random() < 0.8:
                conn.weight += random.uniform(-weight_perturbation, weight_perturbation)
            else:
                conn.weight = random.uniform(-1, 1)
        
        if random.random() < add_connection_prob:
            self._mutate_add_connection(genome)
        
        if random.random() < add_node_prob:
            self._mutate_add_node(genome)
    
    def _mutate_add_connection(self, genome: Genome) -> None:
        potential_in_nodes = [node for node in genome.nodes.values() if node.type != 'output']
        potential_out_nodes = [node for node in genome.nodes.values() if node.type != 'input']
        
        in_node = random.choice(potential_in_nodes)
        out_node = random.choice(potential_out_nodes)

        for conn in genome.connections:
            if conn.in_node.id == in_node.id and conn.out_node.id == out_node.id:
                return
        new_weight = random.uniform(-1, 1)
        new_conn = Connection(in_node, out_node, new_weight, enabled=True, innovation_number=0)

        genome.add_connection(new_conn)

    def _mutate_add_node(self, genome: Genome) -> None:
        enabled_conns = [conn for conn in genome.connections if conn.enabled]
        if not enabled_conns:
            return
        conn_to_split = random.choice(enabled_conns)
        conn_to_split.enabled = False
        
        new_node_id = max(genome.nodes.keys()) + 1
        new_node = Node(new_node_id, 'hidden')
        genome.add_node(new_node)

        conn1 = Connection(conn_to_split.in_node, new_node, weight = 1.0, enabled=True, innovation_number=0)
        conn2 = Connection(new_node, conn_to_split.out_node, weight = conn_to_split.weight, enabled=True, innovation_number=0)

        genome.add_connection(conn1)
        genome.add_connection(conn2)
        
    def reproduce(self) -> None:
        
        new_genomes: list[Genome] = []

        best_genome = max(self.genomes, key=lambda g: g.fitness)
        new_genomes.append(best_genome)
        
        while len(new_genomes) < self.population_size:
            parent1, parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_genomes.append(child)
        self.genome = new_genomes
        self.generation += 1