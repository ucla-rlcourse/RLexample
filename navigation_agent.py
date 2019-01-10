# install AI2THOR at https://github.com/allenai/ai2thor
import ai2thor.controller
import random
import matplotlib.pyplot as plt
import time
controller = ai2thor.controller.Controller()
controller.start()

# Kitchens:       FloorPlan1 - FloorPlan30
# Living rooms:   FloorPlan201 - FloorPlan230
# Bedrooms:       FloorPlan301 - FloorPlan330
# Bathrooms:      FloorPLan401 - FloorPlan430
room_id = 200 #random.randint(1, 430)
controller.reset('FloorPlan' + str(room_id))

# gridSize specifies the coarseness of the grid that the agent navigates on
controller.step(dict(action='Initialize', gridSize=0.25))
event = controller.step(dict(action='MoveAhead'))

actions = ['MoveAhead','MoveLeft','MoveRight','MoveBack','RotateLeft','RotateRight']

for i in range(200):
    # random action
    action_id = random.randint(0, len(actions)-1)
    event = controller.step(dict(action=actions[action_id]))
    plt.imshow(event.frame)
    time.sleep(0.3)
    print('step%d action:%s'%(i, actions[action_id]))
