def reward_function(params):
    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']

    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # Base reward
    reward = 1e-3  # Minimum reward
    
    if distance_from_center <= marker_1:
      reward = 1.0  # Car is very close to the center
    elif distance_from_center <= marker_2:
      reward = 0.5  # Car is moderately close to the center
    elif distance_from_center <= marker_3:
      reward = 0.1  # Car is farther from the center but still on track
    else:
      reward = 1e-3  # Car is likely off track
    
    # Penalize if too close to the edge (distance_from_border)
    distance_from_border = 0.5 * track_width - distance_from_center
    if distance_from_border < 0.08:  # Too close to the track's edge
        reward *= 0.1  # Penalize heavily for being near the edge

    return float(reward)
