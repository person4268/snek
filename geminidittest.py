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
import math

# Let's assume these modules exist and work as per the API description
# You would have these in separate files: snake.py and snakerenderer.py
# from snake import SnakeGame, Direction
# from snakerenderer import SnakeRenderer

# --- Hyperparameters ---
# You can tune these to change the agent's learning behavior
GRID_SIZE = 11
N_GAMES = 150          # Number of games to run in parallel
LEARNING_RATE = 1e-4
GAMMA = 0.98           # Discount factor for future rewards
ENTROPY_BETA = 0.01
N_STEPS = 32 # Number of steps to collect before an update

# --- Start of New Transformer Model Definition ---

class RotaryEmbedding(nn.Module):
    """
    Implements 2D Rotary Position Embeddings (RoPE).
    This is applied to the queries and keys in the attention mechanism to
    encode positional information of each pixel on the game grid.
    """
    def __init__(self, dim, grid_size):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.register_buffer("freqs_cis", self._precompute_freqs_cis())

    def _precompute_freqs_cis(self):
        freqs = 1.0 / (10000 ** (torch.arange(0, self.dim, 4).float() / self.dim))
        t = torch.arange(self.grid_size)
        freqs_x = torch.outer(t, freqs)
        freqs_y = torch.outer(t, freqs)
        
        freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
        freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
        
        # Reshape for broadcasting: [grid_size, grid_size, dim/2]
        freqs_cis = torch.cat([freqs_cis_x.unsqueeze(1).expand(-1, self.grid_size, -1),
                               freqs_cis_y.unsqueeze(0).expand(self.grid_size, -1, -1)],
                              dim=-1)
        # Flatten to match sequence length: [grid_size*grid_size, dim/2]
        return freqs_cis.reshape(self.grid_size * self.grid_size, -1)

    def forward(self, x):
        # x shape: [batch, seq_len, num_heads, head_dim]
        # freqs_cis shape: [seq_len, head_dim/2]
        x_complex = x.float().reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_complex)
        
        freqs_cis = self.freqs_cis[None, :, None, :] # [1, seq_len, 1, head_dim/2]
        
        x_rotated = x_complex * freqs_cis
        x_out = torch.view_as_real(x_rotated).flatten(3)
        return x_out.type_as(x)

class MultiHeadAttention(nn.Module):
    """
    A standard Multi-Head Self-Attention layer that incorporates 2D RoPE.
    """
    def __init__(self, embed_dim, num_heads, grid_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rope = RotaryEmbedding(self.head_dim, grid_size)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # [3, batch, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply 2D Rotary Positional Embeddings
        q = self.rope(q.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        k = self.rope(k.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.out_proj(output)
        return output

class DiTBlock(nn.Module):
    """
    A single Transformer block, as used in Vision Transformers (and DiT).
    It consists of a Multi-Head Attention layer and a Feed-Forward Network.
    """
    def __init__(self, embed_dim, num_heads, grid_size, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, grid_size)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ActorCritic(nn.Module):
    """
    The Actor-Critic network implemented as a 4-layer Vision Transformer.
    """
    def __init__(self, input_dims, n_actions, embed_dim=64, num_heads=4, depth=2):
        super(ActorCritic, self).__init__()
        
        # --- Model Hyperparameters ---
        channels, height, width = input_dims
        self.grid_size = height
        
        # --- 1. Patch Embedding ---
        # Each pixel (1x1 patch) is treated as a token and projected into the embedding space.
        self.patch_embed = nn.Linear(channels, embed_dim)
        
        # --- 2. Transformer Blocks ---
        # A stack of 4 blocks to process the sequence of pixel-tokens.
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, self.grid_size) for _ in range(depth)
        ])
        
        # --- 3. Final Normalization and Heads ---
        self.norm_out = nn.LayerNorm(embed_dim)
        
        # Actor head: determines the policy
        self.actor = nn.Linear(embed_dim, n_actions)
        
        # Critic head: estimates the state value
        self.critic = nn.Linear(embed_dim, 1)

    def forward(self, state):
        # state shape: [batch, channels, height, width]
        
        # 1. Flatten grid and project pixels into embedding space
        x = state.flatten(2).permute(0, 2, 1) # [batch, height*width, channels]
        x = self.patch_embed(x) # [batch, seq_len, embed_dim]
        
        # 2. Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # 3. Normalize and pool tokens (average pooling)
        x = self.norm_out(x)
        x = x.mean(dim=1) # [batch, embed_dim]
        
        # 4. Get policy and value
        action_logits = self.actor(x)
        action_probs = F.softmax(action_logits, dim=-1)
        state_value = self.critic(x)
        
        return action_probs, state_value

# --- End of New Transformer Model Definition ---


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
        self.entropies = []


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
        entropies = torch.cat(self.entropies).to(self.device)

        # Calculate Advantage: A(s,a) = R - V(s)
        advantage = returns - state_values
        
        # Calculate Actor (policy) loss
        actor_loss = -(log_probs * advantage.detach()).mean()
        
        # Calculate Critic (value) loss
        critic_loss = F.mse_loss(state_values, returns)

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
                reward = -0.05
                
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