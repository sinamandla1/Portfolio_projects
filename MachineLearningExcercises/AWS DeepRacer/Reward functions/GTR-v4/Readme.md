I didnt upload the other models since they were exactly the same ast GTS-v2, they only differ in time trained.

GTR-v4 is an adaptation of the previous models with improvements, including 3hrs of training and additions in the
reward function. The model starts of with a reward of 0 (since I think its better to learn from nothing and maintain a level
of output beneficial to the increase in score). I added a harsh punishment for being too close to the edge. I added a
speed incentive of being higher then 0.09m/s, while considering the distance from center line. This all works after 
satisfying the condition of having all 4 wheels on the track.
