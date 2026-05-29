from grid_adventure.grid import GridState
from grid_adventure.env import ImageObservation
# State steppers
from grid_adventure.step import Action, step
from grid_adventure.grid import to_state
# Utility helpers
import heapq
import time
from collections import deque

# Movements and Objectives are gridstate parameters. For Grid Adventure V1, we will be using the default ones.
from grid_adventure.movements import MOVEMENTS
from grid_adventure.objectives import OBJECTIVES

import torch
import numpy as np
from PIL import Image
from grid_adventure.entities import (
    AgentEntity,
    FloorEntity,
    WallEntity,
    ExitEntity,
    CoinEntity,
    GemEntity,
    KeyEntity,
    LockedDoorEntity,
    LavaEntity,
    BoxEntity,
    SpeedPowerUpEntity,
    ShieldPowerUpEntity,
    PhasingPowerUpEntity,
    create_agent_entity
)

LABEL_MAP = {
    0: FloorEntity, 1: WallEntity, 2: AgentEntity, 3: ExitEntity,
    4: CoinEntity, 5: GemEntity, 6: KeyEntity, 7: LockedDoorEntity,
    8: LavaEntity, 9: BoxEntity, 10: SpeedPowerUpEntity, 11: ShieldPowerUpEntity, 12: PhasingPowerUpEntity
}

def get_model(device: str = "cpu", dtype: str | None = None):
    """
    Return a TorchScript model loaded from an embedded, base64-encoded compressed blob.
    Self-contained: no need for the original Python class.

    Args:
        device: Where to map the model (e.g., "cpu", "cuda", "cuda:0").
        dtype: Optional dtype to convert parameters/buffers to (e.g., "float32", "float16").
    """
    import base64, io, torch
    import zlib as _z; _decomp = _z.decompress
    _raw = _decomp(base64.b64decode(_blob_b64))
    buf = io.BytesIO(_raw)
    m = torch.jit.load(buf, map_location=device)
    if dtype is not None:
        dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        for p in m.parameters():
            p.data = p.data.to(dt)
        for b in m.buffers():
            b.data = b.data.to(dt)
    m.eval()
    return m


class Node:
    r"""Node class for search tree
    Args:
        parent (Node): the parent node of this node in the tree
        act (Action): the action taken from parent to reach this node
        state (State): the state of this node
        cost (float): the path cost of reaching this state
    """
    
    def __init__(
            self, 
            parent,
            act, 
            state):
        self.parent = parent
        self.act = act
        self.state = state
        
    def __str__(self):
        return str(self.state)
    def __lt__(self, _):
        return False  # tiebreaker: treat all nodes as equal priority
    def __eq__(self, node):
        """Compare whether two nodes have the same state"""
        return isinstance(node, Node) and self.state == node.state
    def __hash__(self):
        """Node can be used as a KeyValue"""
        return hash(self.state)

class Agent:
    """Grid Adventure: Variant 1 agent template.

    This class is the single public interface that Coursemology will import and
    interact with when evaluating your submission. You should extend the
    internals (add helper classes / functions in other files if you wish) but
    MUST preserve:

    1. The class name: Agent
    2. The public method: step(self, state: GridState | ImageObservation) -> Action

    High‑level lifecycle per environment tick:
        state  --->  step(...)  --->  Action

    The "state" object type depends on the task:
    - Task 1: A fully structured GridState instance.
    - Task 2: An ImageObservation dictionary whose primary observation is an RGBA image
      plus limited structured metadata in the 'info' sub‑dict. In this case you
      typically perform perception to build (or approximate) an internal
      structured representation before planning.
    - Task 3: Input state could be either a GridState instance 
      or an ImageObservation dictionary

    Constraints:
    - Keep per‑step latency small (single CPU, ~1GB RAM). Avoid O(W*H) scans of
      the full grid every step.
    - Determinism helps reproducibility; seed your own RNG if you add any
      random components.

    You may add __init__ parameters (with defaults) if needed for your own
    development, but the grader will instantiate Agent() with no arguments.
    """

    def __init__(self):
        """Initialize your agent.

        Put all one‑time setup here (e.g., hardcoded ML model weights, 
        precomputing heuristic tables). Keep it fast and memory‑light 
        to respect platform limits.
        """
        # Placeholder for any future initialization logic
        self.route = []
        self.dist = {}
        self.model = get_model()

    def step(self, state: GridState | ImageObservation) -> Action:
        """Return the next action given the current environment state.

        Parameters
        ----------
        state : GridState | ImageObservation
            - If a GridState instance (Tasks 1 and 3): you have direct, structured
              access to grid, entities, objective message, score, etc.
            - If a ImageObservation dict (Tasks 2 and 3): contains 'image' (H×W×4 RGBA uint8)
              plus 'info' sub‑dictionary (agent stats, partial config, message).
              You likely need to parse the image into an internal representation.

        Returns
        -------
        Action
            A valid action from the Action enum. Must always return a member;
            never return None.
        """
        if isinstance(state, GridState):
            if self.route:
                return self.route.pop(0)
            else:
                self.route = self.a_star(state)
                return self.route.pop(0)
        else:
            if self.route:
                return self.route.pop(0)
            else:
                converted_state = self.convert_to_gridstate(state)
                self.route = self.a_star(converted_state)
                return self.route.pop(0)

    def convert_to_gridstate(self, img_obs):
        img = img_obs['image']
        info = img_obs['info']
        
        grid_w = info['config']['width']
        grid_h = info['config']['height']
        
        tile_w = img.shape[1] // grid_w
        tile_h = img.shape[0] // grid_h
        
        tiles = []
        for y in range(grid_h):
            for x in range(grid_w):
                tile = img[y*tile_h:(y+1)*tile_h, x*tile_w:(x+1)*tile_w]
                resized = np.array(Image.fromarray(tile).resize((64, 64)))
                t = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
                tiles.append(t)
        
        with torch.no_grad():
            features = self.model(torch.stack(tiles)).argmax(dim=1)
        grid_state = GridState(width=grid_w, height=grid_h, movement=MOVEMENTS['cardinal'], objective=OBJECTIVES['collect_gems_and_exit'], seed=0)
        i = 0
        for y in range(grid_h):
            for x in range(grid_w):
                tile_type = LABEL_MAP[features[i].item()]
                if tile_type == FloorEntity:
                    grid_state.add((x, y), tile_type())
                elif tile_type == AgentEntity:
                    health = info['agent']['health']['current_health']
                    grid_state.add((x, y), create_agent_entity(health=health))
                else:
                    grid_state.add((x, y), FloorEntity())
                    grid_state.add((x, y), tile_type())
                i += 1
        return grid_state
                    
    def search_key(self, state):
        ## Custom key because using the immutable state includes things like turn numbers, so not good to be a key
        return (
            state.position,
            state.inventory,
            state.health,
            state.blocking,
            state.pushable,
            state.collectible,
            state.locked,
            state.requirable,
            state.rewardable,
            state.status,
            state.dead,
            state.immunity,
            state.phasing,
            state.speed,
        )
            
    def a_star(self, state):
        ## Conduct A* to find the best distaance
        pq = [] ## My frontier
        best = {} ## Visited
        
        imut_state = to_state(state)
        self.find_dist(imut_state)
        
        curr = Node(
            parent = None,
            act = None,
            state = imut_state
        )

        root = self.search_key(imut_state)
        best[root] = imut_state.score
        heapq.heappush(pq, (self.h_func(imut_state), curr))
        
        best_win_node = None
        best_win_score = float('-inf')
        start_time = time.perf_counter()
        ## my hard cut 9 secs
        TIME_BUDGET = 9.0 
        
        while pq:
            if time.perf_counter() - start_time > TIME_BUDGET:
                break
            f_val, curr = heapq.heappop(pq)
            
            ## Skip if we've already found a better path to this configuration
            sk = self.search_key(curr.state)
            if curr.state.score < best.get(sk, float('-inf')):
                continue
            
            ## End if nothing can beat the best
            if best_win_node is not None and -f_val <= best_win_score:
                break
            
            ## If win, record it but keep searching for a better one
            if curr.state.win:
                if curr.state.score > best_win_score:
                    best_win_score = curr.state.score
                    best_win_node = curr
                continue
                
            for child in self.transition(curr.state, curr):
                child_key = self.search_key(child.state)
                child_score = child.state.score
                
                ## Only explore if this is the best score we've seen for this configuration
                if child_score <= best.get(child_key, float('-inf')):
                    continue
                    
                best[child_key] = child_score
                heapq.heappush(pq, (-(child_score) + self.h_func(child.state), child))
        
        ## Reverse path from best win node
        if best_win_node is not None:
            backwards = []
            curr = best_win_node
            while curr.parent != None:
                backwards.append(curr.act)
                curr = curr.parent
            backwards.reverse()
            return backwards
        return []
    
    
    def h_func(self, state):
        MULT = 1.5
        agent_id = next(iter(state.agent.keys()))
        agent_pos = state.position.get(agent_id)
        pos = (agent_pos.x, agent_pos.y)
        
        ## Check for powerups
        is_phase = self.get_phasing_turns(state) > 0
        
        ## Get uncollected gem positions
        uncollected = []
        for gem_id in state.requirable:
            gem_pos = state.position.get(gem_id)
            if gem_pos:
                uncollected.append((gem_pos.x, gem_pos.y))
        
        ## All gems collected, just return distance to exit * MULT
        if len(uncollected) == 0:
            to_exit = self.lookup_dist('exit', pos, is_phase)
            return to_exit * MULT
        else:
            ## For each uncollected gem, compute (dist_to_gem + dist_gem_to_exit), take the max
            hv = 0
            for gem in uncollected:
                to_gem = self.lookup_dist(gem, pos, is_phase)
                to_exit = self.lookup_dist('exit', gem, is_phase)
                hv = max(hv, (to_gem + to_exit) * MULT)
            ## Add PICK_UP cost per uncollected gem (3 per gem)
            return hv + len(uncollected) * 3

    def get_phasing_turns(self, state):
        ## Return how many turns of phasing the agent currently has active
        phase_turns = 0
        for eid in state.phasing:
            tl = state.time_limit.get(eid)
            if tl:
                phase_turns = max(phase_turns, tl.amount)
        return phase_turns
    
    def get_speed_turns(self, state):
        ## Return how many turns of speed the agent currently has active
        speed_turns = 0
        for eid in state.speed:
            tl = state.time_limit.get(eid)
            if tl:
                speed_turns = max(speed_turns, tl.amount)
        return speed_turns
    
    def lookup_dist(self, key, pos, has_phasing):
        ## Look up distance, using Manhattan distance if agent has ghost active
        d_normal = self.dist.get(key, {}).get(pos, 999)
        if has_phasing:
            src = self._exit_pos if key == 'exit' else key
            d_phase = abs(src[0] - pos[0]) + abs(src[1] - pos[1])
            return min(d_normal, d_phase)
        return d_normal
    
    def _tiles_to_turns(self, tiles, speed_turns_left):
        ## Convert tile distance to turns, accounting for speed boots (2 tiles/turn)
        if tiles <= 0:
            return 0
        if speed_turns_left <= 0:
            return tiles
        speed_tiles = min(tiles, speed_turns_left * 2)
        speed_used_turns = (speed_tiles + 1) // 2
        remaining_tiles = tiles - speed_tiles
        return speed_used_turns + remaining_tiles
    
    def find_dist(self, state):
        ## Find distance reachable distance from each coin, gem and exit. Ignore locked doors and movable objects. Use BFS
        w = state.width
        h = state.height
        
        walls = set()
        
        for eid in state.blocking:
            ## Taking the walls only
            if eid not in state.locked and eid not in state.pushable:
                pos = state.position.get(eid)
                if pos:
                    walls.add((pos.x, pos.y))
                    
        # BFS from exit
        exit_id = next(iter(state.exit.keys()))
        exit_pos = state.position.get(exit_id)
        self.dist['exit'] = self.bfs((exit_pos.x, exit_pos.y), walls, w, h)
        self._exit_pos = (exit_pos.x, exit_pos.y)

        
        # BFS from each gem
        self.gem_positions = []
        for gem_id in list(state.requirable.keys()):
            gem_pos = state.position.get(gem_id)
            if gem_pos:
                gp = (gem_pos.x, gem_pos.y)
                self.gem_positions.append(gp)
                self.dist[gp] = self.bfs(gp, walls, w, h)
        
        ## BFS from each coin
        self._coin_positions = {}
        self._initial_coin_ids = list(state.rewardable.keys())
        for coin_id in list(state.rewardable.keys()):
            coin_pos = state.position.get(coin_id)
            if coin_pos:
                cp = (coin_pos.x, coin_pos.y)
                self._coin_positions[coin_id] = cp
                if cp not in self.dist:
                    self.dist[cp] = self.bfs(cp, walls, w, h)
        
    def bfs(self, start, walls, w, h):
        ## Peforms bfs from the start positio and returns a dict of all reachable positions and their distance from the start. Ignores walls and locked doors
        d = {start: 0}
        q = deque([start])
        while q:
            x, y = q.popleft()
            for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if 0 <= nx < w and 0 <= ny < h and (nx,ny) not in walls and (nx,ny) not in d:
                    d[(nx,ny)] = d[(x,y)] + 1
                    q.append((nx,ny))
        return d
    
    def transition(self, state, curr_node):
        childrens = []
        for action in Action:
            if action == Action.WAIT:
                continue
            next_state = step(state, action)
            if next_state.lose:
                continue
            else:
                new_node = Node(
                    parent = curr_node,
                    act = action,
                    state = next_state
                )
                childrens.append(new_node)
        return childrens

    def parse(self, observation: ImageObservation) -> GridState:
        """Parse image observation into GridState representation.

        NOTE: This method is optional and intended for debugging in Grid Play only. 
        You do not need to implement it for Coursemology submission, but it can be 
        helpful for visualizing your agent's perception during development. 
        Implementing this method will not affect grading.

        Parameters
        ----------
        observation : ImageObservation
            The raw image observation and metadata from the environment.

        Returns
        -------
        GridState
            The reconstructed internal representation of the environment state.
        """
        # Placeholder: implement perception logic here
        pass
    
    def info(self) -> dict[str, str]:
        """Return info about the agent.

        NOTE: This method is optional and intended for debugging in Grid Play only. 
        You do not need to implement it for Coursemology submission, but it can be 
        helpful for visualizing your agent's internal state during development. 
        Implementing this method will not affect grading.
        """
        # Optional: return info about the agent
        return {"name": "AI Agent"}