import pygame
import gym
from gym import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    """
    A grid-world environment following OpenAI Gym conventions with obstacles.
    
    The grid is of size (grid_size x grid_size). The agent starts at the top-left corner,
    the goal is at the bottom-right corner, and there are obstacles that act as a bad terminal state.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size=10):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        
        # Define the action space: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT.
        self.action_space = spaces.Discrete(4)
        
        # Define the observation space: agent's position as (x, y) coordinates.
        self.observation_space = spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)
        
        # Initialize the starting position of the agent and the goal position.
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([grid_size - 1, grid_size - 1])
        
        # Define obstacles: a list of positions that will terminate the episode with a negative reward.
        # You can modify these positions or generate them dynamically.
        self.obstacles = [np.array([3, 3]), np.array([6, 4]), np.array([14, 3])]
        
        # Limit the maximum steps to avoid infinite loops.
        self.max_steps = grid_size * grid_size
        self.current_steps = 0
        
        # Pygame rendering settings.
        self.window_size = 500  # Window size in pixels.
        self.cell_size = self.window_size // self.grid_size  # Size of each grid cell.
        self.screen = None  # This will be initialized in render().

    def reset(self):
        """
        Reset the environment to the initial state.
        Returns:
            The initial observation (agent's starting position).
        """
        self.agent_pos = np.array([0, 0])
        self.current_steps = 0
        return self.agent_pos.copy()

    def step(self, action):
        """
        Apply an action and update the environment's state.
        
        Args:
            action (int): The action to take (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT).
        
        Returns:
            observation (np.array): The new state (agent's position).
            reward (float): The reward obtained after the action.
            done (bool): Whether the episode is finished.
            info (dict): Additional information (empty in this case).
        """
        self.current_steps += 1
        
        # Move the agent according to the action, ensuring it stays within grid bounds.
        if action == 0:  # UP
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 1:  # RIGHT
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)
        elif action == 2:  # DOWN
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 3:  # LEFT
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        
        # Check for collision with obstacles.
        if any((self.agent_pos == obs).all() for obs in self.obstacles):
            reward = -1  # Negative reward for hitting an obstacle.
            done = True  # End the episode immediately.
        # Check if the agent has reached the goal.
        elif np.array_equal(self.agent_pos, self.goal_pos):
            reward = 1  # Positive reward for reaching the goal.
            done = True
        # Check if maximum steps have been exceeded.
        elif self.current_steps >= self.max_steps:
            reward = -0.1  # Small penalty for running out of time.
            done = True
        else:
            reward = -0.1  # Small penalty to encourage efficiency.
            done = False
        
        info = {}
        return self.agent_pos.copy(), reward, done, info

    def render(self, mode="human"):
        """
        Render the grid-world using pygame.
        """
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        
        # Fill the background with white.
        self.screen.fill((255, 255, 255))
        
        # Draw the grid lines.
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
        
        # Draw obstacles in gray.
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs[0] * self.cell_size, obs[1] * self.cell_size,
                                   self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (128, 128, 128), obs_rect)
        
        # Draw the goal cell in green.
        goal_rect = pygame.Rect(self.goal_pos[0] * self.cell_size, self.goal_pos[1] * self.cell_size,
                                self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), goal_rect)
        
        # Draw the agent cell in red.
        agent_rect = pygame.Rect(self.agent_pos[0] * self.cell_size, self.agent_pos[1] * self.cell_size,
                                 self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), agent_rect)
        
        pygame.display.flip()
        
        # Process pygame events to allow window closure.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        return None

    def close(self):
        """
        Clean up the pygame environment.
        """
        if self.screen is not None:
            pygame.quit()
            self.screen = None

if __name__ == "__main__":
    # Initialize pygame at the start to avoid video initialization errors.
    pygame.init()
    
    env = GridWorldEnv(grid_size=10)
    state = env.reset()
    done = False
    clock = pygame.time.Clock()

    while not done:
        # Process pygame events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # Sample a random action from the environment's action space.
        action = env.action_space.sample()
        
        # Take a step in the environment.
        state, reward, done, info = env.step(action)
        print("State:", state, "Reward:", reward, "Done:", done)
        
        # Render the updated state.
        env.render()
        
        # Limit the loop to 5 frames per second.
        clock.tick(5)

    env.close()
