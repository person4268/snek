
import snake
import snakerenderer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import collections
import time

# Let's assume these modules exist and work as per the API description
# You would have these in separate files: snake.py and snakerenderer.py
# from snake import SnakeGame, Direction
# from snakerenderer import SnakeRenderer

# --- Hyperparameters ---
# You can tune these to change the agent's learning behavior
GRID_SIZE = 11
N_GAMES = 150          # Number of games to run in parallel
LEARNING_RATE = 8e-4
GAMMA = 0.995           # Discount factor for future rewards
ENTROPY_BETA = 0.01      # <--- ADD THIS LINE
N_STEPS = 4 # <--- ADD THIS: Number of steps to collect before an update

# --- The Actor-Critic Network ---
# This single network has two outputs:
# 1. Policy (for the Actor): A probability distribution over actions.
# 2. Value (for the Critic): An estimate of the current state's value.

class ActorCritic(nn.Module):
    """The neural network for our agent."""
    def __init__(self, input_dims, n_actions):
        super(ActorCritic, self).__init__()

        # Shared convolutional base to process the grid
        self.shared_base = nn.Sequential(
            nn.Conv2d(input_dims[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Actor head: determines the policy
        self.actor = nn.Sequential(
            nn.Linear(32 * input_dims[1] * input_dims[2], 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_actions) # Outputs logits for each action
        )

        # Critic head: estimates the state value
        self.critic = nn.Sequential(
            nn.Linear(32 * input_dims[1] * input_dims[2], 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1) # Outputs a single value
        )

    def forward(self, state):
        """A forward pass through the network."""
        # The state is a tensor of shape [batch_size, channels, height, width]
        base_out = self.shared_base(state)
        
        # Flatten the output for the linear layers
        base_out = base_out.view(base_out.size(0), -1)

        # Get action probabilities (policy) from the actor head
        action_logits = self.actor(base_out)
        action_probs = F.softmax(action_logits, dim=-1)

        # Get the state value from the critic head
        state_value = self.critic(base_out)

        return action_probs, state_value


# --- The Agent ---
# The agent encapsulates the model and the learning logic.

class Agent:
    """The A2C agent that learns to play Snake."""
    def __init__(self, input_dims, n_actions):
        self.gamma = GAMMA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = ActorCritic(input_dims, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # We will store trajectories from our parallel games here
        self.log_probs = []
        self.state_values = []
        self.rewards = []
        self.dones = []
        self.entropies = [] # <--- ADD THIS LINE


    def choose_actions(self, states):
        """Chooses actions for a batch of states."""
        # Convert states (numpy array) to a tensor.
        states = torch.from_numpy(states).float().to(self.device)
        
        # The forward pass MUST be done with gradient tracking enabled
        # so we can backpropagate later in the learn() step.
        action_probs, state_values = self.model(states)
            
        # Create a categorical distribution and sample actions for the whole batch
        dist = Categorical(action_probs)
        actions = dist.sample()

        # Store the entire batch of log probabilities and state values.
        # These tensors are now correctly attached to the computation graph.
        self.log_probs.append(dist.log_prob(actions))
        self.state_values.append(state_values)
        self.entropies.append(dist.entropy())

        return actions.cpu().numpy()
    def store_transition(self, reward, done):
        """Stores the result of a step."""
        self.rewards.append(reward)
        self.dones.append(done)

    def learn(self, next_states):
        """Updates the Actor and Critic networks using vectorized operations."""
        # Convert the final states of our trajectories to a tensor
        next_states = torch.from_numpy(next_states).float().to(self.device)
        
        # Get the value of the final states from the critic
        with torch.no_grad():
            _, next_values = self.model(next_states)
        
        # The rewards and dones are lists of numpy arrays. We'll process them with numpy.
        rewards_arr = np.array(self.rewards)
        dones_arr = np.array(self.dones)
        
        # Prepare a placeholder for the calculated returns
        returns = np.zeros_like(rewards_arr)
        
        # Start with the critic's value of the state AFTER the last action
        discounted_return = next_values.squeeze().cpu().numpy()
        
        # Calculate discounted returns, working backwards from the last step
        for i in reversed(range(len(rewards_arr))):
            reward = rewards_arr[i]
            done = dones_arr[i]
            
            discounted_return = reward + self.gamma * discounted_return * (1 - done)
            returns[i] = discounted_return
            
        # --- Convert all data to tensors for loss calculation ---
        returns = torch.from_numpy(returns.flatten()).float().to(self.device)
        log_probs = torch.cat(self.log_probs).to(self.device)
        state_values = torch.cat(self.state_values).view(-1).to(self.device)
        entropies = torch.cat(self.entropies).to(self.device) # <--- GET ENTROPIES

        # Calculate Advantage: A(s,a) = R - V(s)
        advantage = returns - state_values
        
        # Calculate Actor (policy) loss
        actor_loss = -(log_probs * advantage.detach()).mean()
        
        # Calculate Critic (value) loss
        critic_loss = F.mse_loss(state_values, returns)

        # --- THIS IS THE KEY CHANGE ---
        # Combine losses, subtracting the entropy bonus to encourage exploration
        total_loss = actor_loss + critic_loss - ENTROPY_BETA * entropies.mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()
        
        # Clear memory for the next batch
        self.log_probs, self.state_values, self.rewards, self.dones, self.entropies = [], [], [], [], []
        
        return total_loss.item(), grad_norm, actor_loss.item(), critic_loss.item(), ENTROPY_BETA * entropies.mean()

if __name__ == '__main__':
    # Initialize game environments and renderer
    games = [snake.SnakeGame() for _ in range(N_GAMES)]
    renderer = snakerenderer.SnakeRenderer(min(N_GAMES, 16))
    
    # Initialize agent
    input_dims = (1, GRID_SIZE, GRID_SIZE)
    n_actions = 4
    agent = Agent(input_dims, n_actions)
    
    # --- Training Variables ---
    total_episodes = 0
    scores = collections.deque(maxlen=400)
    
    # Helper function for Manhattan distance
    def get_dist(game):
        if not game.snake: return 0
        return abs(game.snake[0].x - game.food.x) + abs(game.snake[0].y - game.food.y)

    # Track distances for reward shaping
    old_distances = [get_dist(g) for g in games]

    print("Starting training... 🚀")
    start_time = time.time()
    
    while True:
        # Collect a trajectory of N_STEPS
        for step in range(N_STEPS):
            states = np.array([game.get_state() for game in games])
            states = states[:, np.newaxis, :, :] # Add channel dimension
            
            actions = agent.choose_actions(states)

            batch_rewards = []
            batch_dones = []
            
            for i, game in enumerate(games):
                # --- API REVERTED ---
                # 1. Change direction based on agent's action
                game.change_direction(snake.Direction(actions[i]))
                # 2. Tick the game forward
                game_over, food_collected, score = game.tick()
                
                # --- COMBINED REWARD LOGIC ---
                new_dist = get_dist(game)
                
                # 1. Base penalty for staying alive to encourage efficiency
                reward = -0.2
                
                # 2. Add shaping reward for moving towards/away from food
                # if new_dist < old_distances[i]:
                #     reward += 0.1
                # elif new_dist > old_distances[i]:
                #     reward += -0.2
                
                # 3. Override with large terminal rewards for key events
                if food_collected:
                    reward += 10
                elif game_over:
                    reward = -10
                
                old_distances[i] = new_dist
                
                batch_rewards.append(reward)
                batch_dones.append(game_over)
                
                if game_over:
                    scores.append(score)
                    total_episodes += 1
                    game.reset()
                    old_distances[i] = get_dist(game)
            
            agent.store_transition(np.array(batch_rewards), np.array(batch_dones))

        # Perform a learning step after collecting the trajectory
        next_states = np.array([game.get_state() for game in games])
        next_states = next_states[:, np.newaxis, :, :]
        loss, grad_norm, actor_loss, critic_loss, entropy_loss = agent.learn(next_states)

        # Logging and Visualization
        if total_episodes > 0:
            avg_score = np.mean(scores) if scores else 0
            if total_episodes % 100 < N_GAMES:
                print(f"Episodes: {total_episodes} | Avg Score: {avg_score:.2f} | Loss: {loss:.4f} | Time: {time.time() - start_time:.2f}s | Grad Norm: {grad_norm:.4f} | Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f} | Entropy Loss: {entropy_loss:.4f}")
        
        renderer.render(games[:16])