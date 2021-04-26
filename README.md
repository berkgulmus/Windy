# Windy
Simple 2D python game environment for reinforcement learning 


--------Game Environment--------

This class basically handles everything that's game related except for visualising it. Game itself is pretty basic. Player moves the ball left & right, trying to avoid the boxes that are coming from above. There are two different types of boxes, one of them spawns on top of the player (boxF, colour is black when visualized) and the other is spawned randomly(boxR, colour is random as well). Boxes survives hitting the ground and disappear after a certain amount of steps has passed.

The way the game handles collision is rather primitive as well. We have a function that (given a box's position & size and the balls position & size) checks if the ball and the box collide with each other. We use this function to check if they collide after we update their locations. Although one can see easily that this can miss some collisions that happens while the two are moving. To overcome this we also use the function at the halfway points of their movements. This still is not a perfect solution obviously, however we believe it is good enough for this kind of a game.

As observation we return the ball's x position & speed, boxR's x & y position and its remaining life, boxF's x & y position and its remaining life. We scale all of these before returning as well.


































