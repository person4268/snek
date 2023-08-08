REW_ALIVE = 0.1 # per tick
REW_FOOD = 5 # per point
PEN_DEATH = 5 # deducted from reward when game over
PEN_TIMEOUT = 5
MAX_TICKS_ALIVE = 300 # game over if snake is alive for this many ticks


frames_per_batch=1000
total_frames=1_000_000


# yeah idk what half of these do really so we're stealing them from pytorch docs
#lr = 3e-4
lr = 1e-3
max_grad_norm = 1.0

sub_batch_size = 4  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimisation steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4