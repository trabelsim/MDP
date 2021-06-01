#Agent learns the unknown world
#known: GAMMA param.
import mdptoolbox
class Agent:
    v=[] #actual v status
    v_next = [] # next v status

    def print_v(self):
        print(self.v)