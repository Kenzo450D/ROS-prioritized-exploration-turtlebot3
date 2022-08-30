# -- get the ground truth label
def get_gt_cc_env(x_coord, y_coord):
    if abs(x_coord) > 3.2:
        return "small_room"
    elif abs(y_coord) > 3.2: 
        return "small_room"
    if abs(x_coord) < 1.6 and abs(y_coord) < 1.6:
        return "large_room"
    else:
        return "corridor"
    
def get_gt_branched_corridor_env(x_coord, y_coord):
    # -- small room
    if abs(y_coord) > 6.45:
        return "small_room"
    if x_coord < -2.9:
        if abs(y_coord) < 0.8:
            return "corridor"
        else:
            return "small_room"
    elif x_coord > -1.3:
        if y_coord < 4.9 and y_coord > 1:
            return "large_room"
        elif y_coord > -4.9 and y_coord < -1.075:
            return "large_room"
        elif y_coord >= - 1.0725 and y_coord <= 1:
            return "small_room"
        else:
            return "corridor"
    else:
        return "corridor"

def get_gt_straight_corridor_env(x_coord, y_coord):
    # -- large room
    if x_coord > 0:
        return "large_room"
    else:
        if abs(y_coord) < 0.8:
            return "corridor"
        else:
            return "small_room"
