import random
import sys, re

#TODO Implement reading world from file
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

def check_world_input(element):
    global world_size
    if "W" in element:
        digits = [int(s) for s in element.split() if s.isdigit()]
        world_size.append(digits[0])
        world_size.append(digits[1])



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
        uncertainty_distribution.append(digits[0])
        uncertainty_distribution.append(digits[1])
        uncertainty_distribution.append(digits[2])
        sum = 0
        for dig in digits:
            sum+=float(dig)
        if not(sum >= 0 and sum <= 1):
            sys.stderr.write("Error in providing the probability values")



def check_reward(element):
    global reward
    if "R" in element:
        digits = re.findall(r"[-+]+\d*\.\d+|\d+",element)
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
        terminal_states[f"X{terminal_index}"] = digits[0]
        terminal_states[f"Y{terminal_index}"] = digits[1]
        terminal_states[f"rew{terminal_index}"] = digits[2]
        terminal_index+=1


def check_special(element):
    global special_states
    global special_index
    if "B" in element:
        digits = re.findall(r"[-\d]+",element)
        special_states[f"X{special_index}"] = digits[0]
        special_states[f"Y{special_index}"] = digits[1]
        special_states[f"rew{special_index}"] = digits[2]
        special_index+=1


def check_forbidden(element):
    global forbidden_states
    global forbidden_index
    if "F" in element:
        digits = re.findall(r"[-\d]+",element)
        forbidden_states[f"X{forbidden_index}"] = digits[0]
        forbidden_states[f"Y{forbidden_index}"] = digits[1]
        forbidden_index+=1


def read_file(input_arguments):
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