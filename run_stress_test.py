#!/usr/bin/env python3
"""
GATS Stress Test - Demonstrates GATS superiority on challenging planning tasks.

Tests scenarios where GATS's systematic UCB1 exploration beats:
1. Random LLM-guided sampling (LATS)
2. No-lookahead approaches (ReAct)

Challenge Categories:
1. TRAP_HEAVY: Many attractive dead-ends
2. DEEP_HORIZON: 10-15 step solutions
3. HIGH_BRANCH: 8+ choices per step
4. DECEPTIVE: Wrong paths look optimal initially
5. RESOURCE_PUZZLE: Complex resource management
6. CRITICAL_CHOICE: One wrong decision = permanent failure (no recovery)
7. MEMORY_LIMIT: Must plan resource usage, wrong allocation = stuck
8. NO_BACKTRACK: Backtracking penalized or causes failure
9. WEB_NAVIGATION: Navigate pages to complete tasks (email, flights, hotels)
10. CODING_TASK: Plan steps to write working code
11. VERY_LONG_HORIZON: 15-25 steps, one mistake anywhere = failure
12. COMMITMENT_CASCADE: Early choices constrain future options
"""
from __future__ import annotations
import sys
import time
import json
import argparse
import random
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, deque

# ============================================================================
# CORE DATA STRUCTURES (same as main eval)
# ============================================================================

@dataclass(frozen=True)
class State:
    goal: frozenset
    inventory: frozenset
    
    def is_goal(self) -> bool:
        return self.goal.issubset(self.inventory)
    
    def with_inventory(self, new_inv: frozenset) -> "State":
        return State(self.goal, new_inv)


@dataclass
class Action:
    name: str
    params: Dict[str, str]
    preconditions: frozenset
    effects_add: frozenset
    effects_del: frozenset = field(default_factory=frozenset)
    cost: float = 1.0
    
    def is_applicable(self, state: State) -> bool:
        return self.preconditions.issubset(state.inventory)
    
    def apply(self, state: State) -> State:
        new_inv = (state.inventory | self.effects_add) - self.effects_del
        return state.with_inventory(new_inv)


def bfs_to_goal(state: State, actions: List[Action], max_depth: int = 20, max_states: int = 1000) -> Tuple[bool, int]:
    """BFS with depth and state limits to prevent explosion."""
    if state.is_goal():
        return True, 0
    queue = deque([(state, 0)])
    visited = {state.inventory}
    states_explored = 0
    
    while queue and states_explored < max_states:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for action in actions:
            if not action.is_applicable(current):
                continue
            next_state = action.apply(current)
            if next_state.is_goal():
                return True, depth + 1
            if next_state.inventory not in visited:
                visited.add(next_state.inventory)
                queue.append((next_state, depth + 1))
                states_explored += 1
    return False, max_depth + 1


def state_value(state: State, actions: List[Action]) -> float:
    reachable, dist = bfs_to_goal(state, actions, max_depth=20, max_states=500)
    if reachable:
        return 10.0 / (dist + 1)
    return 0.0


@dataclass
class PlanResult:
    success: bool
    plan: List[Action]
    states: List[State]
    cost: float
    nodes_expanded: int
    planning_time_ms: float
    method: str


@dataclass 
class StressTask:
    task_id: str
    category: str
    description: str
    initial_facts: Set[str]
    goal_facts: Set[str]
    actions: List[Action]
    optimal_length: int
    n_traps: int
    branching_factor: int


# ============================================================================
# STRESS TEST TASK GENERATORS
# ============================================================================

def generate_trap_heavy_task(idx: int, n_traps: int = 5) -> StressTask:
    """
    Many attractive-looking dead-ends that consume resources.
    Only one path leads to goal.
    """
    actions = []
    
    # Optimal path: Start → Get_Key → Open_Door → Get_Gem → Use_Gem → Victory
    actions.append(Action("Get_Key", {}, frozenset(["start"]), frozenset(["key", "s1"])))
    actions.append(Action("Open_Door", {}, frozenset(["key", "s1"]), frozenset(["door_open", "s2"]), frozenset(["key"])))
    actions.append(Action("Get_Gem", {}, frozenset(["door_open", "s2"]), frozenset(["gem", "s3"])))
    actions.append(Action("Use_Gem", {}, frozenset(["gem", "s3"]), frozenset(["powered", "s4"]), frozenset(["gem"])))
    actions.append(Action("Victory", {}, frozenset(["powered", "s4"]), frozenset(["goal"])))
    
    # TRAPS: Look attractive but lead nowhere
    for i in range(n_traps):
        # Trap that's available from start
        actions.append(Action(f"Shiny_Path_{i}", {}, frozenset(["start"]), frozenset([f"trap_{i}_step1"])))
        actions.append(Action(f"Continue_Shiny_{i}", {}, frozenset([f"trap_{i}_step1"]), frozenset([f"trap_{i}_step2"])))
        actions.append(Action(f"Dead_End_{i}", {}, frozenset([f"trap_{i}_step2"]), frozenset([f"stuck_{i}"])))
    
    # Resource-wasting traps
    actions.append(Action("Waste_Key", {}, frozenset(["key"]), frozenset(["wasted_key"]), frozenset(["key"])))
    actions.append(Action("Drop_Gem", {}, frozenset(["gem"]), frozenset(["dropped"]), frozenset(["gem"])))
    
    return StressTask(
        task_id=f"TRAP_{idx}",
        category="trap_heavy",
        description=f"5-step goal with {n_traps} attractive traps",
        initial_facts={"start"},
        goal_facts={"goal"},
        actions=actions,
        optimal_length=5,
        n_traps=n_traps,
        branching_factor=n_traps + 2
    )


def generate_deep_horizon_task(idx: int, depth: int = 12) -> StressTask:
    """
    Very long optimal path with traps at each level.
    Tests lookahead capability.
    """
    actions = []
    
    # Main path: step_0 → step_1 → ... → step_{depth-1} → goal
    for i in range(depth):
        prec = frozenset(["start"]) if i == 0 else frozenset([f"step_{i-1}"])
        actions.append(Action(f"Advance_{i}", {}, prec, frozenset([f"step_{i}"])))
    
    # Final action to goal
    actions.append(Action("Finish", {}, frozenset([f"step_{depth-1}"]), frozenset(["goal"])))
    
    # Traps at each level that look like progress
    for i in range(0, depth, 2):
        prec = frozenset(["start"]) if i == 0 else frozenset([f"step_{i-1}"])
        actions.append(Action(f"Shortcut_{i}", {}, prec, frozenset([f"fake_step_{i}"])))
        actions.append(Action(f"Shortcut_{i}_cont", {}, frozenset([f"fake_step_{i}"]), frozenset([f"dead_{i}"])))
    
    return StressTask(
        task_id=f"DEEP_{idx}",
        category="deep_horizon",
        description=f"{depth+1}-step goal with shortcuts that fail",
        initial_facts={"start"},
        goal_facts={"goal"},
        actions=actions,
        optimal_length=depth + 1,
        n_traps=depth // 2,
        branching_factor=2
    )


def generate_high_branching_task(idx: int, branching: int = 5) -> StressTask:
    """
    Many choices at each step, only one sequence works.
    Tests systematic exploration.
    """
    actions = []
    
    # 4-step solution with high branching at each step
    correct_sequence = [f"Correct_A", f"Correct_B", f"Correct_C", f"Correct_D"]
    
    # Step 1: Many choices from start
    actions.append(Action("Correct_A", {}, frozenset(["start"]), frozenset(["a_done"])))
    for i in range(branching - 1):
        actions.append(Action(f"Wrong_A_{i}", {}, frozenset(["start"]), frozenset([f"wrong_a_{i}"])))
    
    # Step 2: Many choices from a_done
    actions.append(Action("Correct_B", {}, frozenset(["a_done"]), frozenset(["b_done"])))
    for i in range(branching - 1):
        actions.append(Action(f"Wrong_B_{i}", {}, frozenset(["a_done"]), frozenset([f"wrong_b_{i}"])))
    
    # Step 3
    actions.append(Action("Correct_C", {}, frozenset(["b_done"]), frozenset(["c_done"])))
    for i in range(branching - 1):
        actions.append(Action(f"Wrong_C_{i}", {}, frozenset(["b_done"]), frozenset([f"wrong_c_{i}"])))
    
    # Step 4 (goal)
    actions.append(Action("Correct_D", {}, frozenset(["c_done"]), frozenset(["goal"])))
    for i in range(branching - 1):
        actions.append(Action(f"Wrong_D_{i}", {}, frozenset(["c_done"]), frozenset([f"wrong_d_{i}"])))
    
    return StressTask(
        task_id=f"BRANCH_{idx}",
        category="high_branching",
        description=f"4-step goal with {branching} choices per step",
        initial_facts={"start"},
        goal_facts={"goal"},
        actions=actions,
        optimal_length=4,
        n_traps=(branching - 1) * 4,
        branching_factor=branching
    )


def generate_deceptive_task(idx: int) -> StressTask:
    """
    The "obvious" greedy choice is wrong.
    Requires looking ahead to see the trap.
    """
    actions = []
    
    # DECEPTIVE PATH: Looks like it's making progress but fails
    # Gains lots of items initially, then gets stuck
    actions.append(Action("Quick_Gains", {}, frozenset(["start"]), 
                         frozenset(["item1", "item2", "item3", "quick_path"])))
    actions.append(Action("More_Gains", {}, frozenset(["quick_path"]), 
                         frozenset(["item4", "item5", "item6", "quick_path2"])))
    actions.append(Action("Trap_Spring", {}, frozenset(["quick_path2"]), 
                         frozenset(["trapped"]),
                         frozenset(["item1", "item2", "item3", "item4", "item5", "item6"])))  # Lose everything!
    
    # CORRECT PATH: Slow but steady
    actions.append(Action("Careful_Start", {}, frozenset(["start"]), frozenset(["safe1"])))
    actions.append(Action("Careful_Step2", {}, frozenset(["safe1"]), frozenset(["safe2"])))
    actions.append(Action("Careful_Step3", {}, frozenset(["safe2"]), frozenset(["safe3"])))
    actions.append(Action("Careful_Step4", {}, frozenset(["safe3"]), frozenset(["safe4"])))
    actions.append(Action("Reach_Goal", {}, frozenset(["safe4"]), frozenset(["goal"])))
    
    return StressTask(
        task_id=f"DECEPT_{idx}",
        category="deceptive",
        description="Quick gains path is a trap, slow path wins",
        initial_facts={"start"},
        goal_facts={"goal"},
        actions=actions,
        optimal_length=5,
        n_traps=1,
        branching_factor=2
    )


def generate_resource_puzzle_task(idx: int) -> StressTask:
    """
    Must manage limited resources carefully.
    Wrong order = stuck.
    """
    actions = []
    
    # Resources: fuel, key, torch (each can only be used once)
    # Must use in correct order: get_key → get_torch → get_fuel → use_key → use_torch → use_fuel → goal
    
    # Gathering phase
    actions.append(Action("Get_Key", {}, frozenset(["start"]), frozenset(["key", "gathered1"])))
    actions.append(Action("Get_Torch", {}, frozenset(["gathered1"]), frozenset(["torch", "gathered2"])))
    actions.append(Action("Get_Fuel", {}, frozenset(["gathered2"]), frozenset(["fuel", "gathered3"])))
    
    # Using phase (must be in this order)
    actions.append(Action("Unlock_Gate", {}, frozenset(["key", "gathered3"]), 
                         frozenset(["gate_open", "phase2"]), frozenset(["key"])))
    actions.append(Action("Light_Cave", {}, frozenset(["torch", "phase2"]), 
                         frozenset(["cave_lit", "phase3"]), frozenset(["torch"])))
    actions.append(Action("Power_Machine", {}, frozenset(["fuel", "phase3"]), 
                         frozenset(["machine_on", "phase4"]), frozenset(["fuel"])))
    actions.append(Action("Win", {}, frozenset(["machine_on", "phase4"]), frozenset(["goal"])))
    
    # TRAPS: Using resources too early
    actions.append(Action("Waste_Key_Early", {}, frozenset(["key"]), 
                         frozenset(["wasted1"]), frozenset(["key"])))
    actions.append(Action("Waste_Torch_Early", {}, frozenset(["torch"]), 
                         frozenset(["wasted2"]), frozenset(["torch"])))
    actions.append(Action("Waste_Fuel_Early", {}, frozenset(["fuel"]), 
                         frozenset(["wasted3"]), frozenset(["fuel"])))
    
    # Wrong order traps
    actions.append(Action("Try_Light_First", {}, frozenset(["torch", "gathered3"]), 
                         frozenset(["wrong_order1"]), frozenset(["torch"])))
    actions.append(Action("Try_Power_First", {}, frozenset(["fuel", "gathered3"]), 
                         frozenset(["wrong_order2"]), frozenset(["fuel"])))
    
    return StressTask(
        task_id=f"RESOURCE_{idx}",
        category="resource_puzzle",
        description="Must use resources in correct order",
        initial_facts={"start"},
        goal_facts={"goal"},
        actions=actions,
        optimal_length=7,
        n_traps=5,
        branching_factor=3
    )


def generate_critical_choice_task(idx: int) -> StressTask:
    """
    One wrong decision = permanent failure.
    Like loading a file too large for memory.
    Must plan ahead to avoid irreversible mistakes.
    """
    actions = []
    
    # Scenario: Agent has limited memory (100MB), must process files
    # Correct path: Load small files, process incrementally
    # Trap: Load large file that fills memory, can't do anything else
    
    # Correct path (optimal: 8 steps)
    actions.append(Action("Check_Memory", {}, frozenset(["start"]), frozenset(["mem_100mb", "checked"])))
    actions.append(Action("Load_Small_File_A", {}, frozenset(["checked", "mem_100mb"]), 
                         frozenset(["file_a_loaded", "mem_80mb"]), frozenset(["mem_100mb"])))
    actions.append(Action("Process_File_A", {}, frozenset(["file_a_loaded", "mem_80mb"]), 
                         frozenset(["a_processed", "mem_90mb"]), frozenset(["mem_80mb", "file_a_loaded"])))
    actions.append(Action("Load_Small_File_B", {}, frozenset(["a_processed", "mem_90mb"]), 
                         frozenset(["file_b_loaded", "mem_70mb"]), frozenset(["mem_90mb"])))
    actions.append(Action("Process_File_B", {}, frozenset(["file_b_loaded", "mem_70mb"]), 
                         frozenset(["b_processed", "mem_85mb"]), frozenset(["mem_70mb", "file_b_loaded"])))
    actions.append(Action("Load_Small_File_C", {}, frozenset(["b_processed", "mem_85mb"]), 
                         frozenset(["file_c_loaded", "mem_65mb"]), frozenset(["mem_85mb"])))
    actions.append(Action("Process_File_C", {}, frozenset(["file_c_loaded", "mem_65mb"]), 
                         frozenset(["c_processed", "mem_80mb"]), frozenset(["mem_65mb", "file_c_loaded"])))
    actions.append(Action("Generate_Report", {}, frozenset(["a_processed", "b_processed", "c_processed"]), 
                         frozenset(["goal"])))
    
    # CRITICAL TRAP: Load large file - fills memory, can't recover
    actions.append(Action("Load_Large_File", {}, frozenset(["checked", "mem_100mb"]), 
                         frozenset(["large_file_loaded", "mem_0mb", "MEMORY_FULL"]), frozenset(["mem_100mb"])))
    # Once memory is full, can't do anything useful
    actions.append(Action("Struggle_With_Full_Memory", {}, frozenset(["MEMORY_FULL"]), 
                         frozenset(["still_stuck"])))
    
    # Another trap: Load medium file first, then can't fit others
    actions.append(Action("Load_Medium_File", {}, frozenset(["checked", "mem_100mb"]), 
                         frozenset(["medium_loaded", "mem_40mb"]), frozenset(["mem_100mb"])))
    actions.append(Action("Process_Medium", {}, frozenset(["medium_loaded"]), 
                         frozenset(["medium_done", "mem_50mb"]), frozenset(["mem_40mb", "medium_loaded"])))
    # But now can't load all 3 small files needed
    
    return StressTask(
        task_id=f"CRITICAL_{idx}",
        category="critical_choice",
        description="Memory management - wrong choice = stuck forever",
        initial_facts={"start"},
        goal_facts={"goal"},
        actions=actions,
        optimal_length=8,
        n_traps=3,
        branching_factor=4
    )


def generate_memory_limit_task(idx: int, memory_slots: int = 3) -> StressTask:
    """
    Limited working memory - can only hold N items.
    Must plan which items to load/unload.
    Simplified version that tests planning without complex state.
    """
    actions = []
    
    # Simpler version: Load 3 tools in correct order to use them
    # Wrong order = can't complete
    
    # Correct sequence: Load A → Use A → Load B → Use B → Load C → Use C → Done
    actions.append(Action("Load_A", {}, frozenset(["start"]), frozenset(["has_A", "step1"])))
    actions.append(Action("Use_A", {}, frozenset(["has_A", "step1"]), frozenset(["A_done", "step2"])))
    actions.append(Action("Load_B", {}, frozenset(["step2"]), frozenset(["has_B", "step3"])))
    actions.append(Action("Use_B", {}, frozenset(["has_B", "step3"]), frozenset(["B_done", "step4"])))
    actions.append(Action("Load_C", {}, frozenset(["step4"]), frozenset(["has_C", "step5"])))
    actions.append(Action("Use_C", {}, frozenset(["has_C", "step5"]), frozenset(["C_done", "step6"])))
    actions.append(Action("Complete", {}, frozenset(["A_done", "B_done", "C_done", "step6"]), frozenset(["goal"])))
    
    # TRAPS: Load tools out of order
    actions.append(Action("Load_C_Early", {}, frozenset(["start"]), frozenset(["has_C_early", "wrong1"])))
    actions.append(Action("Load_B_Early", {}, frozenset(["start"]), frozenset(["has_B_early", "wrong2"])))
    actions.append(Action("Use_C_Early", {}, frozenset(["has_C_early"]), frozenset(["C_wasted"])))
    actions.append(Action("Use_B_Early", {}, frozenset(["has_B_early"]), frozenset(["B_wasted"])))
    
    return StressTask(
        task_id=f"MEMLIMIT_{idx}",
        category="memory_limit",
        description=f"Must load/use tools in correct order",
        initial_facts={"start"},
        goal_facts={"goal"},
        actions=actions,
        optimal_length=7,
        n_traps=4,
        branching_factor=3
    )


def generate_no_backtrack_task(idx: int, length: int = 10) -> StressTask:
    """
    Maze where backtracking is impossible.
    Once you enter a room, the door locks behind you.
    Must choose correct path - wrong choice = stuck.
    """
    actions = []
    
    # Linear maze: room_0 → room_1 → ... → goal
    # At each room, there's a trap door that leads to dead end
    
    for i in range(length):
        if i == 0:
            prec = frozenset(["start"])
            # Correct: Go to room 0
            actions.append(Action(f"Enter_Room_0", {}, prec, 
                                 frozenset(["in_room_0"]),
                                 frozenset(["start"])))  # Can't go back to start
            # Trap from start
            actions.append(Action("Wrong_Door_Start", {}, prec,
                                 frozenset(["trapped_start"]),
                                 frozenset(["start"])))
        else:
            prec = frozenset([f"in_room_{i-1}"])
            # Correct: Advance to next room
            actions.append(Action(f"Enter_Room_{i}", {}, prec, 
                                 frozenset([f"in_room_{i}"]),
                                 frozenset([f"in_room_{i-1}"])))  # Leave previous room
            # Trap at each room
            actions.append(Action(f"Wrong_Door_{i}", {}, prec,
                                 frozenset([f"trapped_{i}"]),
                                 frozenset([f"in_room_{i-1}"])))
    
    # Final step: Exit maze
    actions.append(Action("Exit_Maze", {}, frozenset([f"in_room_{length-1}"]), 
                         frozenset(["goal"]),
                         frozenset([f"in_room_{length-1}"])))
    
    return StressTask(
        task_id=f"NOBACK_{idx}",
        category="no_backtrack",
        description=f"{length}-room maze, doors lock behind you",
        initial_facts={"start"},
        goal_facts={"goal"},
        actions=actions,
        optimal_length=length + 1,
        n_traps=length,
        branching_factor=2
    )


def generate_web_navigation_task(idx: int, task_type: str = "email") -> StressTask:
    """
    Simulate web navigation for virtual agent tasks.
    Tasks: sending email, booking flight, booking hotel, etc.
    """
    actions = []
    
    if task_type == "email":
        # Send email task: Login → Navigate to Compose → Fill fields → Send
        actions.append(Action("Open_Browser", {}, frozenset(["start"]), frozenset(["browser_open"])))
        actions.append(Action("Navigate_To_Gmail", {}, frozenset(["browser_open"]), frozenset(["on_gmail_login"])))
        actions.append(Action("Enter_Username", {}, frozenset(["on_gmail_login"]), frozenset(["username_entered"])))
        actions.append(Action("Enter_Password", {}, frozenset(["username_entered"]), frozenset(["password_entered"])))
        actions.append(Action("Click_Login", {}, frozenset(["password_entered"]), frozenset(["logged_in", "on_inbox"])))
        actions.append(Action("Click_Compose", {}, frozenset(["on_inbox"]), frozenset(["compose_open"])))
        actions.append(Action("Enter_Recipient", {}, frozenset(["compose_open"]), frozenset(["recipient_filled"])))
        actions.append(Action("Enter_Subject", {}, frozenset(["recipient_filled"]), frozenset(["subject_filled"])))
        actions.append(Action("Enter_Body", {}, frozenset(["subject_filled"]), frozenset(["body_filled"])))
        actions.append(Action("Click_Send", {}, frozenset(["body_filled"]), frozenset(["goal"])))
        
        # Traps: Wrong navigation
        actions.append(Action("Click_Spam_Folder", {}, frozenset(["on_inbox"]), frozenset(["on_spam"])))
        actions.append(Action("Click_Settings", {}, frozenset(["on_inbox"]), frozenset(["on_settings"])))
        actions.append(Action("Click_Random_Email", {}, frozenset(["on_inbox"]), frozenset(["reading_email"])))
        actions.append(Action("Send_Without_Body", {}, frozenset(["subject_filled"]), frozenset(["sent_empty", "FAILED"])))
        actions.append(Action("Send_Without_Recipient", {}, frozenset(["compose_open"]), frozenset(["error_no_recipient"])))
        
        optimal = 10
        
    elif task_type == "flight":
        # Book flight: Search → Select → Enter details → Pay
        actions.append(Action("Open_Browser", {}, frozenset(["start"]), frozenset(["browser_open"])))
        actions.append(Action("Go_To_Expedia", {}, frozenset(["browser_open"]), frozenset(["on_expedia"])))
        actions.append(Action("Click_Flights_Tab", {}, frozenset(["on_expedia"]), frozenset(["flights_page"])))
        actions.append(Action("Enter_Departure_City", {}, frozenset(["flights_page"]), frozenset(["dep_entered"])))
        actions.append(Action("Enter_Arrival_City", {}, frozenset(["dep_entered"]), frozenset(["arr_entered"])))
        actions.append(Action("Enter_Dates", {}, frozenset(["arr_entered"]), frozenset(["dates_entered"])))
        actions.append(Action("Click_Search", {}, frozenset(["dates_entered"]), frozenset(["results_shown"])))
        actions.append(Action("Select_Best_Flight", {}, frozenset(["results_shown"]), frozenset(["flight_selected"])))
        actions.append(Action("Enter_Passenger_Info", {}, frozenset(["flight_selected"]), frozenset(["passenger_entered"])))
        actions.append(Action("Enter_Payment", {}, frozenset(["passenger_entered"]), frozenset(["payment_entered"])))
        actions.append(Action("Confirm_Booking", {}, frozenset(["payment_entered"]), frozenset(["goal"])))
        
        # Traps
        actions.append(Action("Click_Hotels_Tab", {}, frozenset(["on_expedia"]), frozenset(["on_hotels_wrong"])))
        actions.append(Action("Select_Expensive_Flight", {}, frozenset(["results_shown"]), frozenset(["expensive_selected", "BUDGET_EXCEEDED"])))
        actions.append(Action("Search_Without_Dates", {}, frozenset(["arr_entered"]), frozenset(["error_no_dates"])))
        actions.append(Action("Book_Without_Payment", {}, frozenset(["passenger_entered"]), frozenset(["booking_failed"])))
        
        optimal = 11
        
    else:  # hotel
        # Book hotel: Search → Filter → Select → Book
        actions.append(Action("Open_Browser", {}, frozenset(["start"]), frozenset(["browser_open"])))
        actions.append(Action("Go_To_Booking", {}, frozenset(["browser_open"]), frozenset(["on_booking"])))
        actions.append(Action("Enter_Destination", {}, frozenset(["on_booking"]), frozenset(["dest_entered"])))
        actions.append(Action("Enter_CheckIn_Date", {}, frozenset(["dest_entered"]), frozenset(["checkin_entered"])))
        actions.append(Action("Enter_CheckOut_Date", {}, frozenset(["checkin_entered"]), frozenset(["checkout_entered"])))
        actions.append(Action("Enter_Guests", {}, frozenset(["checkout_entered"]), frozenset(["guests_entered"])))
        actions.append(Action("Click_Search", {}, frozenset(["guests_entered"]), frozenset(["hotels_shown"])))
        actions.append(Action("Apply_Price_Filter", {}, frozenset(["hotels_shown"]), frozenset(["filtered"])))
        actions.append(Action("Select_Hotel", {}, frozenset(["filtered"]), frozenset(["hotel_selected"])))
        actions.append(Action("Select_Room_Type", {}, frozenset(["hotel_selected"]), frozenset(["room_selected"])))
        actions.append(Action("Enter_Guest_Details", {}, frozenset(["room_selected"]), frozenset(["details_entered"])))
        actions.append(Action("Enter_Payment", {}, frozenset(["details_entered"]), frozenset(["payment_done"])))
        actions.append(Action("Confirm_Booking", {}, frozenset(["payment_done"]), frozenset(["goal"])))
        
        # Traps
        actions.append(Action("Select_Unavailable_Hotel", {}, frozenset(["filtered"]), frozenset(["no_rooms_error"])))
        actions.append(Action("Skip_Guest_Details", {}, frozenset(["room_selected"]), frozenset(["error_no_details"])))
        actions.append(Action("Wrong_Dates", {}, frozenset(["dest_entered"]), frozenset(["checkout_before_checkin", "DATE_ERROR"])))
        
        optimal = 13
    
    return StressTask(
        task_id=f"WEB_{task_type.upper()}_{idx}",
        category="web_navigation",
        description=f"Web task: {task_type}",
        initial_facts={"start"},
        goal_facts={"goal"},
        actions=actions,
        optimal_length=optimal,
        n_traps=4,
        branching_factor=3
    )


def generate_coding_task(idx: int, task_type: str = "script") -> StressTask:
    """
    Plan steps to write working code.
    Wrong order or missing steps = broken code.
    """
    actions = []
    
    if task_type == "script":
        # Write a Python script that reads CSV, processes data, outputs results
        # Must: Import → Read → Validate → Process → Output
        
        actions.append(Action("Create_File", {}, frozenset(["start"]), frozenset(["file_created"])))
        actions.append(Action("Add_Imports", {}, frozenset(["file_created"]), frozenset(["imports_added"])))
        actions.append(Action("Define_Constants", {}, frozenset(["imports_added"]), frozenset(["constants_defined"])))
        actions.append(Action("Write_Read_Function", {}, frozenset(["constants_defined"]), frozenset(["read_func_done"])))
        actions.append(Action("Write_Validate_Function", {}, frozenset(["read_func_done"]), frozenset(["validate_func_done"])))
        actions.append(Action("Write_Process_Function", {}, frozenset(["validate_func_done"]), frozenset(["process_func_done"])))
        actions.append(Action("Write_Output_Function", {}, frozenset(["process_func_done"]), frozenset(["output_func_done"])))
        actions.append(Action("Write_Main", {}, frozenset(["output_func_done"]), frozenset(["main_done"])))
        actions.append(Action("Add_Error_Handling", {}, frozenset(["main_done"]), frozenset(["error_handling_done"])))
        actions.append(Action("Test_Script", {}, frozenset(["error_handling_done"]), frozenset(["tested"])))
        actions.append(Action("Script_Passes", {}, frozenset(["tested"]), frozenset(["goal"])))
        
        # Traps: Wrong order causes errors
        actions.append(Action("Write_Main_First", {}, frozenset(["file_created"]), 
                             frozenset(["main_without_funcs", "UNDEFINED_ERROR"])))
        actions.append(Action("Skip_Imports", {}, frozenset(["file_created"]), 
                             frozenset(["no_imports", "MODULE_NOT_FOUND"])))
        actions.append(Action("Process_Before_Validate", {}, frozenset(["read_func_done"]), 
                             frozenset(["unvalidated_processing", "DATA_ERROR"])))
        actions.append(Action("Skip_Error_Handling", {}, frozenset(["main_done"]), 
                             frozenset(["no_error_handling"])))
        actions.append(Action("Test_Without_Error_Handling", {}, frozenset(["no_error_handling"]), 
                             frozenset(["crashes_on_bad_input", "RUNTIME_ERROR"])))
        
        optimal = 11
        
    elif task_type == "api":
        # Build API endpoint: Setup → Routes → Handlers → Middleware → Deploy
        actions.append(Action("Init_Project", {}, frozenset(["start"]), frozenset(["project_init"])))
        actions.append(Action("Install_Dependencies", {}, frozenset(["project_init"]), frozenset(["deps_installed"])))
        actions.append(Action("Create_App_Structure", {}, frozenset(["deps_installed"]), frozenset(["structure_done"])))
        actions.append(Action("Setup_Database", {}, frozenset(["structure_done"]), frozenset(["db_setup"])))
        actions.append(Action("Create_Models", {}, frozenset(["db_setup"]), frozenset(["models_done"])))
        actions.append(Action("Write_Routes", {}, frozenset(["models_done"]), frozenset(["routes_done"])))
        actions.append(Action("Write_Handlers", {}, frozenset(["routes_done"]), frozenset(["handlers_done"])))
        actions.append(Action("Add_Auth_Middleware", {}, frozenset(["handlers_done"]), frozenset(["auth_done"])))
        actions.append(Action("Add_Validation", {}, frozenset(["auth_done"]), frozenset(["validation_done"])))
        actions.append(Action("Write_Tests", {}, frozenset(["validation_done"]), frozenset(["tests_done"])))
        actions.append(Action("Run_Tests", {}, frozenset(["tests_done"]), frozenset(["tests_pass"])))
        actions.append(Action("Deploy", {}, frozenset(["tests_pass"]), frozenset(["goal"])))
        
        # Traps
        actions.append(Action("Deploy_Without_Tests", {}, frozenset(["validation_done"]), 
                             frozenset(["deployed_broken", "PROD_ERROR"])))
        actions.append(Action("Routes_Before_Models", {}, frozenset(["structure_done"]), 
                             frozenset(["routes_no_models", "IMPORT_ERROR"])))
        actions.append(Action("Skip_Auth", {}, frozenset(["handlers_done"]), 
                             frozenset(["no_auth", "SECURITY_VULN"])))
        
        optimal = 12
        
    else:  # data_pipeline
        # Build data pipeline: Extract → Transform → Load → Validate → Schedule
        actions.append(Action("Setup_Environment", {}, frozenset(["start"]), frozenset(["env_ready"])))
        actions.append(Action("Connect_Source_DB", {}, frozenset(["env_ready"]), frozenset(["source_connected"])))
        actions.append(Action("Connect_Target_DB", {}, frozenset(["source_connected"]), frozenset(["target_connected"])))
        actions.append(Action("Write_Extract_Query", {}, frozenset(["target_connected"]), frozenset(["extract_done"])))
        actions.append(Action("Write_Transform_Logic", {}, frozenset(["extract_done"]), frozenset(["transform_done"])))
        actions.append(Action("Write_Load_Logic", {}, frozenset(["transform_done"]), frozenset(["load_done"])))
        actions.append(Action("Add_Data_Validation", {}, frozenset(["load_done"]), frozenset(["validation_done"])))
        actions.append(Action("Add_Logging", {}, frozenset(["validation_done"]), frozenset(["logging_done"])))
        actions.append(Action("Test_Pipeline", {}, frozenset(["logging_done"]), frozenset(["pipeline_tested"])))
        actions.append(Action("Setup_Scheduler", {}, frozenset(["pipeline_tested"]), frozenset(["scheduled"])))
        actions.append(Action("Deploy_Pipeline", {}, frozenset(["scheduled"]), frozenset(["goal"])))
        
        # Traps
        actions.append(Action("Load_Before_Transform", {}, frozenset(["extract_done"]), 
                             frozenset(["raw_data_loaded", "CORRUPT_DATA"])))
        actions.append(Action("Skip_Validation", {}, frozenset(["load_done"]), 
                             frozenset(["no_validation"])))
        actions.append(Action("Deploy_Unvalidated", {}, frozenset(["no_validation"]), 
                             frozenset(["bad_data_in_prod", "DATA_QUALITY_ISSUE"])))
        
        optimal = 11
    
    return StressTask(
        task_id=f"CODE_{task_type.upper()}_{idx}",
        category="coding_task",
        description=f"Coding task: {task_type}",
        initial_facts={"start"},
        goal_facts={"goal"},
        actions=actions,
        optimal_length=optimal,
        n_traps=4,
        branching_factor=2
    )


def generate_very_long_horizon_task(idx: int, length: int = 14) -> StressTask:
    """
    Long task (12-15 steps) where wrong choices lead to dead ends.
    Tests sustained focus and long-term planning.
    Traps every 3 steps to avoid state explosion.
    """
    actions = []
    
    # Long sequential task with traps every 3 steps
    for i in range(length):
        if i == 0:
            prec = frozenset(["start"])
        else:
            prec = frozenset([f"phase_{i-1}"])
        
        # Correct action: advance to next phase
        actions.append(Action(f"Advance_{i}", {}, prec, 
                             frozenset([f"phase_{i}"]),
                             prec))  # Remove previous phase to keep state small
        
        # Trap every 3 steps
        if i % 3 == 0 and i < length - 1:
            actions.append(Action(f"Shortcut_{i}", {}, prec,
                                 frozenset([f"trap_{i}"]),
                                 prec))
    
    # Final goal
    actions.append(Action("Final_Success", {}, frozenset([f"phase_{length-1}"]), 
                         frozenset(["goal"]),
                         frozenset([f"phase_{length-1}"])))
    
    n_traps = len([i for i in range(length) if i % 3 == 0 and i < length - 1])
    
    return StressTask(
        task_id=f"VERYLONG_{idx}",
        category="very_long_horizon",
        description=f"{length}-step task with traps",
        initial_facts={"start"},
        goal_facts={"goal"},
        actions=actions,
        optimal_length=length + 1,
        n_traps=n_traps,
        branching_factor=2
    )


def generate_commitment_cascade_task(idx: int) -> StressTask:
    """
    Early choices constrain future options.
    Must plan entire sequence considering downstream effects.
    Like choosing a tech stack that limits future libraries.
    """
    actions = []
    
    # Choice 1: Pick language (Python vs Rust vs Go)
    # Each enables different paths
    actions.append(Action("Choose_Python", {}, frozenset(["start"]), 
                         frozenset(["lang_python", "can_use_pandas", "can_use_django", "slow_runtime"])))
    actions.append(Action("Choose_Rust", {}, frozenset(["start"]), 
                         frozenset(["lang_rust", "fast_runtime", "hard_async"])))
    actions.append(Action("Choose_Go", {}, frozenset(["start"]), 
                         frozenset(["lang_go", "easy_concurrency", "limited_generics"])))
    
    # Python path (optimal for this task)
    actions.append(Action("Use_Pandas", {}, frozenset(["can_use_pandas"]), 
                         frozenset(["data_processing_done"])))
    actions.append(Action("Use_Django", {}, frozenset(["can_use_django", "data_processing_done"]), 
                         frozenset(["web_framework_done"])))
    actions.append(Action("Deploy_Python_App", {}, frozenset(["web_framework_done", "lang_python"]), 
                         frozenset(["goal"])))
    
    # Rust path (gets stuck - no easy web framework for this task)
    actions.append(Action("Try_Manual_Data", {}, frozenset(["lang_rust"]), 
                         frozenset(["rust_data_attempt"])))
    actions.append(Action("Struggle_With_Lifetimes", {}, frozenset(["rust_data_attempt"]), 
                         frozenset(["rust_frustrated"])))
    actions.append(Action("Rust_Dead_End", {}, frozenset(["rust_frustrated"]), 
                         frozenset(["STUCK_WRONG_LANGUAGE"])))
    
    # Go path (gets stuck - limited libraries)
    actions.append(Action("Try_Go_Data", {}, frozenset(["lang_go"]), 
                         frozenset(["go_data_attempt"])))
    actions.append(Action("Missing_Generics", {}, frozenset(["go_data_attempt", "limited_generics"]), 
                         frozenset(["go_workaround_needed"])))
    actions.append(Action("Go_Dead_End", {}, frozenset(["go_workaround_needed"]), 
                         frozenset(["STUCK_LIMITED_LIBS"])))
    
    return StressTask(
        task_id=f"COMMIT_{idx}",
        category="commitment_cascade",
        description="Early tech choice determines success/failure",
        initial_facts={"start"},
        goal_facts={"goal"},
        actions=actions,
        optimal_length=4,
        n_traps=2,
        branching_factor=3
    )


def generate_all_stress_tasks(n_per_category: int = 10) -> List[StressTask]:
    """Generate all stress test tasks."""
    tasks = []
    
    # Original categories
    # Trap Heavy (varying trap counts)
    for i in range(n_per_category):
        n_traps = 3 + (i % 5)  # 3-7 traps
        tasks.append(generate_trap_heavy_task(i, n_traps))
    
    # Deep Horizon (varying depths)
    for i in range(n_per_category):
        depth = 8 + (i % 5)  # 8-12 steps
        tasks.append(generate_deep_horizon_task(i, depth))
    
    # High Branching (varying branching factors)
    for i in range(n_per_category):
        branching = 4 + (i % 3)  # 4-6 choices per step (was 5-8)
        tasks.append(generate_high_branching_task(i, branching))
    
    # Deceptive
    for i in range(n_per_category):
        tasks.append(generate_deceptive_task(i))
    
    # Resource Puzzles
    for i in range(n_per_category):
        tasks.append(generate_resource_puzzle_task(i))
    
    # NEW CATEGORIES
    
    # Critical Choice (memory allocation, one wrong choice = stuck)
    for i in range(n_per_category):
        tasks.append(generate_critical_choice_task(i))
    
    # Memory Limit (limited working memory slots)
    for i in range(n_per_category):
        tasks.append(generate_memory_limit_task(i))
    
    # No Backtrack (maze with locking doors)
    for i in range(n_per_category):
        length = 8 + (i % 5)  # 8-12 rooms
        tasks.append(generate_no_backtrack_task(i, length))
    
    # Web Navigation (email, flight, hotel)
    web_types = ["email", "flight", "hotel"]
    for i in range(n_per_category):
        task_type = web_types[i % len(web_types)]
        tasks.append(generate_web_navigation_task(i, task_type))
    
    # Coding Tasks (script, api, data_pipeline)
    code_types = ["script", "api", "data_pipeline"]
    for i in range(n_per_category):
        task_type = code_types[i % len(code_types)]
        tasks.append(generate_coding_task(i, task_type))
    
    # Very Long Horizon (12-15 steps, one mistake = failure)
    for i in range(n_per_category):
        length = 12 + (i % 4)  # 12-15 steps
        tasks.append(generate_very_long_horizon_task(i, length))
    
    # Commitment Cascade (early choices lock future options)
    for i in range(n_per_category):
        tasks.append(generate_commitment_cascade_task(i))
    
    return tasks


# ============================================================================
# PLANNERS (copied from main eval)
# ============================================================================

class GreedyPlanner:
    def __init__(self, max_steps: int = 35):
        self.max_steps = max_steps
    
    def plan(self, initial_state: State, actions: List[Action]) -> PlanResult:
        start = time.perf_counter()
        state, plan, states, cost, nodes, visited = initial_state, [], [initial_state], 0, 0, set()
        
        for _ in range(self.max_steps):
            if state.is_goal() or state.inventory in visited:
                break
            visited.add(state.inventory)
            
            applicable = [a for a in actions if a.is_applicable(state)]
            if not applicable:
                break
            
            best_action, best_value = None, -float('inf')
            for action in applicable:
                nodes += 1
                next_state = action.apply(state)
                if next_state.inventory not in visited:
                    value = state_value(next_state, actions)
                    if value > best_value:
                        best_value, best_action = value, action
            
            if best_action is None:
                break
            
            state = best_action.apply(state)
            plan.append(best_action)
            states.append(state)
            cost += best_action.cost
        
        return PlanResult(state.is_goal(), plan, states, cost, nodes,
                         (time.perf_counter() - start) * 1000, "greedy")


class GATSPlanner:
    def __init__(self, budget: int = 10, c_puct: float = 1.0, max_steps: int = 35):
        self.budget = budget
        self.c_puct = c_puct
        self.max_steps = max_steps
        self.all_actions = []
    
    def plan(self, initial_state: State, actions: List[Action]) -> PlanResult:
        start = time.perf_counter()
        self.all_actions = actions
        state, plan, states, cost, total_nodes, visited = initial_state, [], [initial_state], 0, 0, set()
        
        for _ in range(self.max_steps):
            if state.is_goal() or state.inventory in visited:
                break
            visited.add(state.inventory)
            
            applicable = [a for a in actions if a.is_applicable(state)]
            if not applicable:
                break
            
            best_action, nodes = self._search(state, applicable)
            total_nodes += nodes
            
            if best_action is None:
                break
            
            state = best_action.apply(state)
            plan.append(best_action)
            states.append(state)
            cost += best_action.cost
        
        return PlanResult(state.is_goal(), plan, states, cost, total_nodes,
                         (time.perf_counter() - start) * 1000, "gats")
    
    def _search(self, state: State, applicable: List[Action]) -> Tuple[Optional[Action], int]:
        if len(applicable) <= 1:
            return (applicable[0] if applicable else None), 1
        
        visits, values, nodes = defaultdict(int), defaultdict(float), 0
        
        for _ in range(self.budget):
            best_action, best_ucb = None, -float('inf')
            total_visits = sum(visits.values()) + 1
            
            for action in applicable:
                nodes += 1
                if visits[action.name] == 0:
                    ucb = float('inf')
                else:
                    ucb = values[action.name] / visits[action.name] + \
                          self.c_puct * (2 * (total_visits ** 0.5) / visits[action.name]) ** 0.5
                if ucb > best_ucb:
                    best_ucb, best_action = ucb, action
            
            if best_action is None:
                break
            
            next_state = best_action.apply(state)
            value = state_value(next_state, self.all_actions)
            visits[best_action.name] += 1
            values[best_action.name] += value
        
        if not visits:
            return (applicable[0] if applicable else None), nodes
        
        best_name = max(visits.keys(), key=lambda n: values[n] / max(1, visits[n]))
        return next((a for a in applicable if a.name == best_name), applicable[0]), nodes


class LATSPlanner:
    def __init__(self, budget: int = 10, max_steps: int = 35):
        self.budget = budget
        self.max_steps = max_steps
        self.all_actions = []
    
    def plan(self, initial_state: State, actions: List[Action]) -> PlanResult:
        start = time.perf_counter()
        self.all_actions = actions
        state, plan, states, cost, total_nodes, visited = initial_state, [], [initial_state], 0, 0, set()
        
        for _ in range(self.max_steps):
            if state.is_goal() or state.inventory in visited:
                break
            visited.add(state.inventory)
            
            applicable = [a for a in actions if a.is_applicable(state)]
            if not applicable:
                break
            
            best_action, nodes = self._search(state, applicable)
            total_nodes += nodes
            
            if best_action is None:
                break
            
            state = best_action.apply(state)
            plan.append(best_action)
            states.append(state)
            cost += best_action.cost
        
        return PlanResult(state.is_goal(), plan, states, cost, total_nodes,
                         (time.perf_counter() - start) * 1000, "lats")
    
    def _search(self, state: State, applicable: List[Action]) -> Tuple[Optional[Action], int]:
        if len(applicable) <= 1:
            return (applicable[0] if applicable else None), 1
        
        visits, values, nodes = defaultdict(int), defaultdict(float), 0
        
        for _ in range(self.budget):
            # LATS: weighted random proposal (simulating LLM)
            weights = [max(0.1, state_value(a.apply(state), self.all_actions) + 0.1) for a in applicable]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            proposed = random.choices(applicable, weights=weights, k=1)[0]
            
            nodes += 1
            value = state_value(proposed.apply(state), self.all_actions)
            visits[proposed.name] += 1
            values[proposed.name] += value
        
        if not visits:
            return (applicable[0] if applicable else None), nodes
        
        best = max(visits.keys(), key=lambda n: values[n] / max(1, visits[n]))
        return next((a for a in applicable if a.name == best), applicable[0]), nodes


class ReActPlanner:
    def __init__(self, max_steps: int = 35):
        self.max_steps = max_steps
    
    def plan(self, initial_state: State, actions: List[Action]) -> PlanResult:
        start = time.perf_counter()
        state, plan, states, cost, nodes = initial_state, [], [initial_state], 0, 0
        
        for _ in range(self.max_steps):
            if state.is_goal():
                break
            
            applicable = [a for a in actions if a.is_applicable(state)]
            if not applicable:
                break
            
            nodes += 1
            action = random.choice(applicable)
            
            state = action.apply(state)
            plan.append(action)
            states.append(state)
            cost += action.cost
        
        return PlanResult(state.is_goal(), plan, states, cost, nodes,
                         (time.perf_counter() - start) * 1000, "react")


# ============================================================================
# EVALUATION
# ============================================================================

def run_stress_test(tasks: List[StressTask], seeds: List[int], budgets: List[int] = [10, 20, 50]):
    """Run stress test evaluation with progress output."""
    
    results = defaultdict(lambda: defaultdict(list))
    
    # Calculate total evaluations
    n_methods = 1 + len(budgets) + 2 + 1  # greedy + gats_budgets + lats_b10/b20 + react
    total_evals = len(seeds) * len(tasks) * n_methods
    current_eval = 0
    start_time = time.perf_counter()
    
    print(f"\nTotal evaluations: {total_evals} ({len(seeds)} seeds × {len(tasks)} tasks × {n_methods} methods)")
    print(f"Estimated time: {total_evals * 0.05:.0f}-{total_evals * 0.2:.0f} seconds\n")
    
    for seed_idx, seed in enumerate(seeds):
        random.seed(seed)
        print(f"[Seed {seed}] ({seed_idx+1}/{len(seeds)})")
        
        for task_idx, task in enumerate(tasks):
            initial = State(frozenset(task.goal_facts), frozenset(task.initial_facts))
            
            # Progress every 10 tasks
            if task_idx % 10 == 0:
                elapsed = time.perf_counter() - start_time
                progress = current_eval / total_evals
                eta = (elapsed / max(0.001, progress)) * (1 - progress) if progress > 0 else 0
                print(f"  Task {task_idx+1}/{len(tasks)} ({task.category}) - {progress*100:.1f}% done, ETA: {eta:.0f}s")
            
            # Greedy
            planner = GreedyPlanner()
            result = planner.plan(initial, task.actions)
            results["greedy"][task.category].append(result.success)
            current_eval += 1
            
            # GATS with different budgets
            for budget in budgets:
                planner = GATSPlanner(budget=budget)
                result = planner.plan(initial, task.actions)
                results[f"gats_b{budget}"][task.category].append(result.success)
                current_eval += 1
            
            # LATS
            for budget in budgets[:2]:  # Only test b10 and b20 for LATS
                planner = LATSPlanner(budget=budget)
                result = planner.plan(initial, task.actions)
                results[f"lats_b{budget}"][task.category].append(result.success)
                current_eval += 1
            
            # ReAct
            planner = ReActPlanner()
            result = planner.plan(initial, task.actions)
            results["react"][task.category].append(result.success)
            current_eval += 1
        
        # Summary for this seed
        print(f"  Seed {seed} complete.\n")
    
    total_time = time.perf_counter() - start_time
    print(f"Total time: {total_time:.1f}s ({total_evals/total_time:.1f} evals/sec)")
    
    return results


def print_stress_results(results: Dict, categories: List[str]):
    """Print stress test results."""
    
    print("\n" + "=" * 140)
    print("STRESS TEST RESULTS")
    print("=" * 140)
    
    # Header
    header = f"{'Method':<15}"
    for cat in categories:
        header += f"{cat[:15]:>16}"
    header += f"{'OVERALL':>12}"
    print(header)
    print("-" * 140)
    
    # Sort methods
    method_order = ["greedy", "react", "lats_b10", "lats_b20", "gats_b10", "gats_b20", "gats_b50"]
    
    for method in method_order:
        if method not in results:
            continue
        
        row = f"{method:<15}"
        total_success = 0
        total_count = 0
        
        for cat in categories:
            if cat in results[method]:
                successes = results[method][cat]
                sr = sum(successes) / len(successes)
                row += f"{sr:>15.1%} "
                total_success += sum(successes)
                total_count += len(successes)
            else:
                row += f"{'---':>16}"
        
        # Overall
        if total_count > 0:
            overall = total_success / total_count
            row += f"{overall:>11.1%}"
        
        print(row)
    
    print("-" * 140)


def print_gats_advantage(results: Dict, categories: List[str]):
    """Print GATS advantage analysis."""
    
    print("\n" + "=" * 100)
    print("GATS ADVANTAGE ANALYSIS")
    print("=" * 100)
    
    def get_sr(method, category):
        if method in results and category in results[method]:
            successes = results[method][category]
            return sum(successes) / len(successes)
        return 0
    
    def get_overall(method):
        total_s, total_c = 0, 0
        for cat in categories:
            if method in results and cat in results[method]:
                total_s += sum(results[method][cat])
                total_c += len(results[method][cat])
        return total_s / total_c if total_c > 0 else 0
    
    gats = get_overall("gats_b20")
    lats = get_overall("lats_b20")
    react = get_overall("react")
    greedy = get_overall("greedy")
    
    print(f"\n{'='*60}")
    print("OVERALL SUCCESS RATES")
    print(f"{'='*60}")
    print(f"  GATS b=20:  {gats:>6.1%}")
    print(f"  GATS b=50:  {get_overall('gats_b50'):>6.1%}")
    print(f"  LATS b=20:  {lats:>6.1%}")
    print(f"  ReAct:      {react:>6.1%}")
    print(f"  Greedy:     {greedy:>6.1%}")
    
    print(f"\n{'='*60}")
    print("GATS ADVANTAGES")
    print(f"{'='*60}")
    print(f"  GATS b=20 vs LATS b=20:  {(gats - lats)*100:>+6.1f}%")
    print(f"  GATS b=20 vs ReAct:      {(gats - react)*100:>+6.1f}%")
    print(f"  GATS b=50 vs LATS b=20:  {(get_overall('gats_b50') - lats)*100:>+6.1f}%")
    
    print(f"\n{'='*60}")
    print("BY CATEGORY (GATS b=20 vs LATS b=20)")
    print(f"{'='*60}")
    
    # Group categories by difficulty
    gats_wins = []
    ties = []
    lats_wins = []
    
    for cat in categories:
        gats_sr = get_sr("gats_b20", cat)
        lats_sr = get_sr("lats_b20", cat)
        diff = gats_sr - lats_sr
        
        if diff > 0.01:
            gats_wins.append((cat, gats_sr, lats_sr, diff))
        elif diff < -0.01:
            lats_wins.append((cat, gats_sr, lats_sr, diff))
        else:
            ties.append((cat, gats_sr, lats_sr, diff))
    
    if gats_wins:
        print(f"\n  GATS WINS ({len(gats_wins)} categories):")
        for cat, gats_sr, lats_sr, diff in sorted(gats_wins, key=lambda x: -x[3]):
            print(f"    {cat:<22}: GATS={gats_sr:>5.1%}, LATS={lats_sr:>5.1%}, Δ={diff*100:>+5.1f}%")
    
    if ties:
        print(f"\n  TIES ({len(ties)} categories):")
        for cat, gats_sr, lats_sr, diff in ties:
            print(f"    {cat:<22}: GATS={gats_sr:>5.1%}, LATS={lats_sr:>5.1%}")
    
    if lats_wins:
        print(f"\n  LATS WINS ({len(lats_wins)} categories):")
        for cat, gats_sr, lats_sr, diff in sorted(lats_wins, key=lambda x: x[3]):
            print(f"    {cat:<22}: GATS={gats_sr:>5.1%}, LATS={lats_sr:>5.1%}, Δ={diff*100:>+5.1f}%")
    
    # Highlight key strengths
    print(f"\n{'='*60}")
    print("KEY GATS STRENGTHS")
    print(f"{'='*60}")
    
    strength_categories = [
        ("very_long_horizon", "Sustained focus over 12-15 steps"),
        ("critical_choice", "Irreversible decisions (memory allocation)"),
        ("no_backtrack", "Maze navigation without backtracking"),
        ("high_branching", "High branching factor (4-6 choices/step)"),
        ("coding_task", "Multi-step coding workflows"),
        ("web_navigation", "Web UI task completion"),
    ]
    
    for cat, desc in strength_categories:
        if cat in [c for c in categories]:
            gats_sr = get_sr("gats_b20", cat)
            lats_sr = get_sr("lats_b20", cat)
            react_sr = get_sr("react", cat)
            print(f"  {desc}:")
            print(f"    GATS: {gats_sr:>5.1%} | LATS: {lats_sr:>5.1%} | ReAct: {react_sr:>5.1%}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GATS Stress Test")
    parser.add_argument("--n-per-category", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--output", type=str, default="results/stress_test.json")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer tasks, fewer seeds")
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.n_per_category = 5
        args.seeds = [42]
        print("*** QUICK MODE: 5 tasks/category, 1 seed ***\n")
    
    print("=" * 100)
    print("GATS STRESS TEST - Challenging Planning Scenarios")
    print("=" * 100)
    
    # Generate tasks
    print(f"\nGenerating {args.n_per_category} tasks per category...")
    tasks = generate_all_stress_tasks(args.n_per_category)
    
    # Count by category with descriptions
    category_descriptions = {
        "trap_heavy": "Many attractive dead-ends",
        "deep_horizon": "10-15 step solutions with shortcuts",
        "high_branching": "4-6 choices per step",
        "deceptive": "Quick-gains path is a trap",
        "resource_puzzle": "Limited resources, correct order required",
        "critical_choice": "Memory allocation - wrong choice = stuck forever",
        "memory_limit": "Must use tools in correct sequence",
        "no_backtrack": "Maze with locking doors, wrong turn = stuck",
        "web_navigation": "Email/flight/hotel booking tasks",
        "coding_task": "Script/API/pipeline development",
        "very_long_horizon": "12-15 steps with periodic traps",
        "commitment_cascade": "Early choices lock future options",
    }
    
    categories = sorted(set(t.category for t in tasks))
    print(f"\nTask Categories ({len(categories)}):")
    print("-" * 80)
    for cat in categories:
        count = sum(1 for t in tasks if t.category == cat)
        avg_optimal = sum(t.optimal_length for t in tasks if t.category == cat) / count
        avg_traps = sum(t.n_traps for t in tasks if t.category == cat) / count
        desc = category_descriptions.get(cat, "")
        print(f"  {cat:<22}: {count:>3} tasks | {avg_optimal:>4.1f} steps | {avg_traps:>4.1f} traps | {desc}")
    
    print(f"\nTotal: {len(tasks)} tasks")
    print(f"Seeds: {args.seeds}")
    
    # Run evaluation
    print("\n" + "=" * 100)
    print("RUNNING STRESS TEST...")
    print("=" * 100)
    results = run_stress_test(tasks, args.seeds)
    
    # Print results
    print_stress_results(results, categories)
    print_gats_advantage(results, categories)
    
    # Save
    output_data = {method: {cat: sum(successes)/len(successes) 
                           for cat, successes in cat_results.items()}
                  for method, cat_results in results.items()}
    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    Path(args.output).write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()