# Grid Adventure: Maze Solver

An AI agent that plays **Grid Adventure**, a tile-based puzzle game. The agent combines a custom A\*-based search algorithm with a trained CNN to navigate mazes, collect gems, and exit — all within a strict time budget.

---

## The Game

Grid Adventure is a grid-based puzzle where the agent starts at a fixed position and must **collect all gems and reach the exit** to win. The level contains a mix of tile types the agent must navigate:

| Entity | Effect |
|---|---|
| Coin | Optional pickup — adds points to score |
| Gem | Required — must collect all gems before the exit unlocks |
| Key | Unlocks locked doors |
| Locked Door | Impassable without a key |
| Lava | Damages the agent on contact |
| Box | Pushable obstacle |
| Speed Boots | Move 2 tiles per turn for a limited duration |
| Shield | Grants temporary immunity to lava damage |
| Phasing | Lets the agent pass through walls for a limited duration |

The scoring rewards collecting coins and gems efficiently while minimising turns taken.

---

## Part 1 — A\* Search

### Overview

The agent plans a full path upfront using a best-first search adapted from classical A\*. Since the goal is to **maximise total score** rather than minimise cost, the priority function becomes:

```
priority = -score + h(state)
```

where `h` estimates the remaining turn cost to complete the level.

### Key Design Choices

**Dominance pruning instead of a closed set.**
The same physical configuration (positions, inventory, health, power-ups) can be reached with different scores depending on which coins were collected along the way. A `best` dictionary tracks the highest score seen per configuration via a custom `search_key` that excludes turn count and score. A child node is only expanded if it strictly improves on the best known score for its configuration.

**Continue past the first win.**
When a winning state is popped from the priority queue, the agent records it but keeps searching. Search terminates early when the optimistic bound `score - h` of the best remaining node can no longer beat the best win found so far — guaranteeing we don't stop at a suboptimal plan.

**Heuristic.**
For each uncollected gem, compute `dist(agent → gem) + dist(gem → exit)` and take the max across all gems. The distance is converted to turns via `_tiles_to_turns`, which accounts for the speed boots power-up (2 tiles per turn while active). The result is multiplied by 3 and `3 × num_gems` is added for pickup actions. Coins are also factored in — each uncollected coin subtracts 5 from the heuristic (`coin_bonus = rem_coins × 5`), making states with coins still available look more attractive to expand and nudging the search toward coin-collecting paths.

**Precomputed BFS distances.**
At search start, BFS is run once from the exit, each gem, and each coin, treating locked doors and pushable boxes as passable. These distances feed the heuristic in O(1) per lookup. When the agent has phasing active, the heuristic falls back to Manhattan distance since walls no longer block movement.

**Hard time budget.**
A 9.5-second cutoff guarantees the agent always returns a plan within the grader's 10s per-grid limit, even if search is incomplete. The best win found so far is returned when time runs out.

---

## Part 2 — CNN Tile Classifier

When the environment provides an image observation instead of a structured state, the agent must **perceive the grid from pixels**. A custom CNN classifies each tile in the image into one of 13 entity types.

### Dataset Generation

Training data was generated programmatically by building a fixed level containing every entity type and rendering it across **500 random seeds**. Each seed produces a visually different rendering (different sprite frames from each entity's animation cycle). Individual tiles were sliced out and labelled by querying the underlying grid state — no manual annotation required.

### Model Architecture

The CNN takes a **64×64 RGBA tile** (4 channels) as input and outputs a 13-class prediction.

```
Conv(4→32) → BatchNorm → LeakyReLU → MaxPool
Conv(32→64) → BatchNorm → LeakyReLU → MaxPool
Conv(64→128) → BatchNorm → LeakyReLU
→ Global Average Pooling
→ Dropout(0.3) → Linear(128→128) → LeakyReLU → Dropout(0.3) → Linear(128→13)
```

### Key Learning Points

**Data augmentation with colour jitter.**
Since the sprites use distinct colour palettes, the model could overfit to exact pixel values. Augmentations like colour jitter during dataset generation improved generalisation across visual variations between seeds.

**Hyperparameter search.**
Multiple combinations of learning rate, batch size, and number of training epochs were tested. Results were tracked by training loss and per-class validation accuracy to find the best configuration.

**Input size and depth experiments.**
Different input resolutions and numbers of convolutional layers were evaluated. Larger inputs captured more detail but increased inference time; deeper networks improved accuracy up to a point before showing diminishing returns.

**Per-class accuracy evaluation.**
Rather than relying on overall accuracy alone, performance was broken down by tile type. This revealed which entity classes were harder to classify (e.g. visually similar tiles) and guided further improvements.

**Global Average Pooling (GAP) instead of flatten.**
GAP averages each feature map into a single value before the fully connected layers. This significantly reduces the parameter count compared to flattening and improves generalisation by making the model less sensitive to exact spatial position.

**BatchNorm + LeakyReLU.**
Batch normalisation stabilises training by normalising activations between layers. LeakyReLU was chosen over standard ReLU to avoid dying neurons, where a ReLU unit can get stuck outputting zero and stop learning.

**Dropout for regularisation.**
Dropout(0.3) is applied before both fully connected layers, randomly zeroing 30% of activations during training. This forces the network to learn redundant representations and reduces overfitting on the training set.

**Adam + StepLR scheduler.**
The Adam optimiser handles adaptive learning rates per parameter. A `StepLR` scheduler decays the learning rate by a factor of 0.1 every 15 epochs, resulting in a smoother loss curve and better final convergence than a fixed learning rate.

**TorchScript + base64 embedding for deployment.**
The trained model is compiled to TorchScript and compressed into a base64-encoded blob embedded directly inside `solver.py`. This makes the submission self-contained — no external weight file is needed, and the model loads entirely from memory at runtime.

---

## Setup

```bash
conda env create -f environment.yml
conda activate maze-solver
```

Requires the `grid-adventure` and `grid-play` packages (installed automatically via the environment file).
