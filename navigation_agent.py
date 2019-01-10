# install AI2THOR at https://github.com/allenai/ai2thor
import ai2thor.controller
import random
import matplotlib.pyplot as plt
import time

# Kitchens:       FloorPlan1 - FloorPlan30
# Living rooms:   FloorPlan201 - FloorPlan230
# Bedrooms:       FloorPlan301 - FloorPlan330
# Bathrooms:      FloorPLan401 - FloorPlan430
room_id = 205
agent_id = 1 # 1 is random agent, 2 is keyboard agent (control by you)

controller = ai2thor.controller.Controller()
controller.start()

controller.reset('FloorPlan' + str(room_id))

# gridSize specifies the coarseness of the grid that the agent navigates on
controller.step(dict(action='Initialize', gridSize=0.25))

actions = ['MoveAhead','MoveLeft','MoveRight','MoveBack','RotateLeft','RotateRight']

for i in range(200):
    if agent_id == 1:
        action_id = random.randint(0, len(actions)-1) # random action
    elif agent_id == 2:
        # input your action
        key = int(input('press 1,2,3,4,5,6\n')) # type 1,2,3,4,5
        action_id = key-1
    event = controller.step(dict(action=actions[action_id]))
    plt.imshow(event.frame)
    time.sleep(0.2)
    print('step%d action:%s' % (i, actions[action_id]))
