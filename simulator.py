import random
import sys, re , mdptoolbox
import numpy as np

total_size_world = 0
world_size = []
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


def initialize_rewards():
    global r_rewards, reward
    #Apply global reward for all the states
    # R_REWARDS [ Y ] [ X ]
    r_rewards = np.ones(shape=(world_size[1], world_size[0]))
    print(r_rewards)
    for y in range(world_size[1]):
        for x in range(world_size[0]):
            r_rewards[y][x] = reward
    #Apply reward only for terminal states from the terminal states dict.
    print(terminal_states)
    for key in terminal_states:
        r_rewards[terminal_states[key][1]-1][terminal_states[key][0]-1] = terminal_states[key][2]

    #Apply special reward for special states
    for key in special_states:
        r_rewards[special_states[key][1]-1][special_states[key][0]-1] = special_states[key][2]

    #TODO Handle the forbidden states inside the grid map
    for key in forbidden_states:
        r_rewards[forbidden_states[key][1]-1][forbidden_states[key][0]-1] = 0




def start_value_iteration():
    global world_size , r_rewards
    v = [[0] * world_size[0]] * world_size[1]  # actual v status
    v_next = [[0] * world_size[0]] * world_size[1]  # next v status
    pi_policy = [[0] * world_size[0]] * world_size[1]
    delta = 0 #checking v - v_next
    iteration = 0
    initialize_rewards()
    print(r_rewards)
    print("-----------")
    print(r_rewards[::-1])

#
# def create_P_matrix():
#     global world_size, total_size_world
#     total_size_world = world_size[0] * world_size[1]
#     print(total_size_world)
#     matrix_P = np.zeros(shape=(len(uncertainty_distribution), total_size_world, total_size_world))
#     # for action in range(len(matrix_P)):
#     #     for state in range(len(matrix_P[1])):
#             # print(matrix_P[state])
#     matrix_R = np.empty((world_size[0],world_size[1]))
#     matrix_R.fill(reward)
#     A = mdptoolbox.mdp.ValueIteration(matrix_P[0],matrix_R,1)
#     print(A)

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
    with open("MDPRL_world1.data","r") as world:
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

