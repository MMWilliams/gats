class TreeOfThoughtAgent:
    """
    Tree-of-Thought baseline.
    
    Differs from GATS:
    - BFS/DFS instead of MCTS
    - LLM-based state evaluation (no formal verification)
    - No world model (direct prompting)
    """
    
    def __init__(self, llm, branching: int = 3, max_depth: int = 5):
        self.llm = llm
        self.branching = branching
        self.max_depth = max_depth
    
    def solve(self, initial_state, task_id):
        # BFS with LLM-scored pruning
        ...