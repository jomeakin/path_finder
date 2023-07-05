# path_finder
A visual representation of the A* path finding algorithm with a graphical user interface that allows users to customise the map.
The algorithm traverses through nodes in an order based upon the sum (f score) of their expected distance from the target node (h score) and the distance to the start node (g score). 
A pandas dataframe is used to record the length of the shortest path to each node that has been considered and what the previous node in the shortest path to it was.
The next node to be considered is the node with the lowest f score and upon considering, its neighbours will be added to the open set containing nodes which are in a queue to be checked.
The visual representation displays nodes which are start, end, barrier, open, closed and path as a specific colour.
If the open set is empty, there is not a possible path from start to end.
