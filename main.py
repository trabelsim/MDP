# NxM world size
# Unique START state S
# Terminal state T
# Walls and prohibited states F
# Special states B

#PARAMETERS:
# N columns
# M rows
# p1 - probability of going into the intended state
# p2 - probability of going to the left of the intended state
# p3 - probability of going to the right of the intended state
# ( 1 - p1 - p2 - p3 ) - probability of going into the opposite side of the intended state

# r - reward for the state


# 1 PART - VALUE ITERATION METHOD
# import mdptoolbox.example
# P, R = mdptoolbox.example.forest()
# vi = mdptoolbox.mdp.ValueIteration(P,R,0.9)
# vi.run()
# print(vi.policy)
#
# md1 = mdptoolbox.mdp.

import sys
import simulator

input_arguments = sys.argv
filenamex=input_arguments[1]
print(f"Loaded world: {filenamex}")

if(len(input_arguments) == 2 and 'data' in input_arguments[1]):
    simulator.read_file(input_arguments, filename=filenamex)
if(len(input_arguments) == 3):
    gammaxx = float(input_arguments[2])
    simulator.read_file(input_arguments, filename=filenamex,gammax=gammaxx)
if(len(input_arguments) == 4):
    gammaxx = float(input_arguments[2])
    explorexx = float(input_arguments[3])
    simulator.read_file(input_arguments,filename=filenamex,gammax=gammaxx,explorex=explorexx)
if len(input_arguments) == 1:
    sys.exit("No data world provided. Exiting.")

simulator.start_value_iteration()
simulator.value_iteration()
simulator.show_results()