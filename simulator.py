import random
import sys, re
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

total_size_world = 0
world_size = []
rows = 0
columns = 0
start_state = []
uncertainty_distribution = []
reward = 0
gamma = 0
explore = 0
terminal_states = {}
terminal_index = 0
special_states = {}
special_index = 0
forbidden_states = {}
forbidden_index = 0

r_rewards = 0
v = 0
v_next = 0
actions = [1,1,1,1]

iterations = 0
plotting_dict = {}

def plot():
    t = list(range(iterations))
    y = plotting_dict['0_1']
    fig, ax = plt.subplots()
    for state in plotting_dict:
        ax.plot(t, plotting_dict[state])
    print(t)
    print(y)
    plt.show()


def initialize_plotting_dict():
    global plotting_dict
    for i in range(rows):
        for j in range(columns):
            key = f"{i}_{j}"
            plotting_dict[key] = []


def add_values_for_plotting(row,col):
    global plotting_dict
    key = f"{row}_{col}"
    plotting_dict[key].append(v_next[row,col])



#Used for updating the value with Ballman equation
def update(row,col):
    global actions, v, v_next
    intended_prob = uncertainty_distribution[0]
    left_miss_prob = uncertainty_distribution[1]
    right_miss_prob = uncertainty_distribution[2]
    back_prob = uncertainty_distribution[3]


    for term_state in terminal_states:
        term_state_x = terminal_states[term_state][0]-1
        term_state_y = terminal_states[term_state][1]-1
        if ( row == term_state_y and col == term_state_x):
            v_next[row][col] = r_rewards[row][col]
            # print(f"terminal state in {term_state_y},{term_state_x}")
            return v_next

    for forb_state in forbidden_states:
        forbidden_x = forbidden_states[forb_state][0] - 1
        forbidden_y = forbidden_states[forb_state][1] - 1
        if ( row == forbidden_y and col == forbidden_x):
            v_next[row][col] = r_rewards[row][col]
            # print(f"forbidden state in {forbidden_y},{forbidden_x}")
            return v_next
    # P(s' | s,a) * V(s')
    actions[0] = intended_prob * go_up(row,col) + left_miss_prob * go_left(row,col) + right_miss_prob * go_right(row,col)
    actions[1] = intended_prob * go_down(row,col) + left_miss_prob * go_left(row,col) + right_miss_prob * go_right(row,col)
    actions[2] = intended_prob * go_left(row,col) + left_miss_prob * go_down(row,col) + right_miss_prob * go_up(row,col)
    actions[3] = intended_prob * go_right(row,col) + left_miss_prob * go_up(row,col) + right_miss_prob * go_down(row,col)

    best_action = find_max_action(actions)
    v_next[row][col] = r_rewards[row][col] + float(gamma)*actions[best_action]
    return v_next



def value_iteration( ):
    global v, v_next
    initialize_plotting_dict()
    delta = 0
    n_iter = 0
    counter=0
    v = np.zeros(shape=(rows, columns))
    v_next = np.zeros(shape=(rows, columns))
    checking = True
    v_next = v
    while checking:
        n_iter+=1
        for row in range(rows):
            # print(f"Row: {row}")
            for col in range(columns):
                # print(f"Column: {col}")
                v_next = update(row,col)
                print(f"V VALUE: {v[row][col]}")
                print(f"V_NEXT VALUE: {v_next[row][col]}")
                add_values_for_plotting(row,col)
                error_diff = abs(v_next[row][col] - v[row][col])
                if(error_diff > delta):
                    delta = error_diff
                    print(f"ERROR: {delta}")
        if not( n_iter < iterations):
            checking=False
    print(v[::-1])
    print(plotting_dict)
    plot()

def find_max_action(actions):
    max_index = 0
    for index in range(len(actions)):
        if actions[index] > actions[max_index]:
            max_index = index
    return max_index


def go_up(y,x):
    global v
    # Y are the rows
    # X are the columns
    for key in forbidden_states:
        forbidden_x = forbidden_states[key][0]-1
        forbidden_y = forbidden_states[key][1]-1
        #In case we are in the last row OR there is a forbidden state in the up slot
        if y == rows-1 or ( x == forbidden_x and y == forbidden_y - 1 ):
            # print(f"FORBIDDEN State in (row={forbidden_y}, col={forbidden_x})!")
            return v[y][x]
    #Otherwise go up
    return v[y+1][x]


def go_down(y,x):
    global v
    for key in forbidden_states:
        forbidden_x = forbidden_states[key][0]-1
        forbidden_y = forbidden_states[key][1]-1
        #In case we are in the first row OR there is a forbidden state in the down slot
        if y == 0 or  ( x == forbidden_x and y == forbidden_y + 1 ):
            return v[y][x]
    #Otherwise go down
    return v[y-1][x]



def go_left(y,x):
    global v
    for key in forbidden_states:
        forbidden_x = forbidden_states[key][0]-1
        forbidden_y = forbidden_states[key][1]-1
        #In case we are in the first column OR there is a forbidden state in the left slot
        if x == 0 or  ( x == forbidden_x + 1 and y == forbidden_y ):
            return v[y][x]
    #Otherwise go left
    return v[y][x-1]


def go_right(y,x):
    global v
    for key in forbidden_states:
        forbidden_x = forbidden_states[key][0]-1
        forbidden_y = forbidden_states[key][1]-1
        #In case we are in the last column OR there is a forbidden state in the right slot
        if x == columns-1 or  ( x == forbidden_x - 1 and y == forbidden_y ):
            return v[y][x]
    #Otherwise go right
    return v[y][x+1]



def initialize_rewards():
    global r_rewards, reward, rows, columns
    #Apply global reward for all the states
    # R_REWARDS [ Y ] [ X ]
    rows = world_size[1]
    columns = world_size[0]
    r_rewards = np.ones(shape=(rows, columns))
    for y in range(rows):
        for x in range(columns):
            r_rewards[y][x] = reward
    #Apply reward only for terminal states from the terminal states dict.
    for key in terminal_states:
        r_rewards[terminal_states[key][1]-1][terminal_states[key][0]-1] = terminal_states[key][2]

    #Apply special reward for special states
    for key in special_states:
        r_rewards[special_states[key][1]-1][special_states[key][0]-1] = special_states[key][2]

    #TODO Handle the forbidden states inside the grid map
    for key in forbidden_states:
        r_rewards[forbidden_states[key][1]-1][forbidden_states[key][0]-1] = 0

#    print(r_rewards)
    #TODO Remember to flip in the end only for representation!
#    r_rewards = np.flipud(r_rewards)
#     print(r_rewards)



def start_value_iteration():
    global world_size , r_rewards
    initialize_rewards()


def check_world_input(element):
    global world_size
    if "W" in element:
        digits = [int(s) for s in element.split() if s.isdigit()]
        world_size.append(int(digits[0]))
        world_size.append(int(digits[1]))



#Checking if optional start state position has been provided
def check_start_state(element):
    global start_state
    if "S" in element:
        digits = [int(s) for s in element.split() if s.isdigit()]
        start_state.append(digits[0])
        start_state.append(digits[1])
    else:
        #If the random values are not equal to any terminal state
        #TODO Add the checking condition for a start state for QLearning
        x = random.randint(1,4)
        y = random.randint(1,3)
        for el in terminal_states:
            while x == terminal_states[el]:
                x = random.random(1,4)



def check_uncertainty_distribution(element):
    global uncertainty_distribution
    if "P" in element:
        digits = re.findall(r"[-+]?\d*\.\d+|\d+",element)
        uncertainty_distribution.append(float(digits[0]))
        uncertainty_distribution.append(float(digits[1]))
        uncertainty_distribution.append(float(digits[2]))
        uncertainty_distribution.append((1 - float(digits[0]) - float(digits[1]) - float(digits[2])))
        sum = 0
        for dig in digits:
            sum+=float(dig)
        if not(sum >= 0 and sum <= 1):
            sys.stderr.write("Error in providing the probability values")



def check_reward(element):
    global reward
    if "R" in element:
        digits = re.findall(r"[-+]+\d*\.\d+|[-+]\d+",element)
        reward = digits[0]



def check_gamma(element, input_arguments):
    global gamma
    if "G" in element:
        digits = re.findall(r"[-+]?\d*\.\d+|\d+",element)
        gamma = digits[0]
    else:
        if len(input_arguments) > 1:
            gamma = input_arguments[1]



def check_explore(element, input_arguments):
    global explore
    if "E" in element:
        digits = re.findall(r"[-+]?\d*\.\d+|\d+",element)
        explore = digits[0]
    else:
        if len(input_arguments) > 2:
            explore = input_arguments[2]


def check_terminal(element):
    global terminal_states
    global terminal_index
    if "T" in element:
        digits = re.findall(r"[-\d]+",element)
        terminal_states[terminal_index] = [int(digits[0]),int(digits[1]), int(digits[2])]
        terminal_index+=1


def check_special(element):
    global special_states
    global special_index
    if "B" in element:
        digits = re.findall(r"[-\d]+",element)
        special_states[special_index] = [int(digits[0]), int(digits[1]), int(digits[2])]
        special_index+=1


def check_forbidden(element):
    global forbidden_states
    global forbidden_index
    if "F" in element:
        digits = re.findall(r"[-\d]+",element)
        # forbidden_states[f"X{forbidden_index}"] = digits[0]
        # forbidden_states[f"Y{forbidden_index}"] = digits[1]
        forbidden_states[forbidden_index] = [int(digits[0]), int(digits[1])]
        forbidden_index+=1


def read_file(input_arguments):
    global iterations
    with open("MDPRL_world0.data","r") as world:
        values = world.readlines()
        for i in range(len(values)):
            print(values[i])
            check_world_input(values[i])
            check_start_state(values[i])
            check_uncertainty_distribution(values[i])
            check_reward(values[i])
            check_gamma(values[i], input_arguments)
            check_explore(values[i], input_arguments)
            check_terminal(values[i])
            check_special(values[i])
            check_forbidden(values[i])
            iterations = 30


def print_values():
    global world_size, start_state, uncertainty_distribution, reward, gamma, explore
    print(f"World size: {world_size}")
    print(f"Start state: {start_state}")
    print(f"Uncertainty distribution: {uncertainty_distribution}")
    print(f"Reward parameter: {reward}")
    print(f"Gamma parameter: {gamma}")
    print(f"Exploration parameter: {explore}")
    print(f"Terminal states: {terminal_states}")
    print(f"Special states: {special_states}")
    print(f"Forbidden states: {forbidden_states}")
#TODO Updates the map by the actions received

#Generates the trial

