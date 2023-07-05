import pygame as pg
from queue import PriorityQueue
import pandas as pd
import math

pg.display.set_caption('Path Finder')

TITLE_IMAGE = pg.image.load('Instructions.png')
WIDTH = 800
WIN = pg.display.set_mode((WIDTH,WIDTH))
RED = (255,0,0)
GREEN = (0,255,0)
WHITE = (255,255,255)
BLACK = (0,0,0)
PURPLE = (128,0,128)
ORANGE = (205,125,0)
BLUE = (0,0,200)
ROWS = 50
NODE_WIDTH = WIDTH/ROWS


class Node:
    """
    Node class for each square in space
    """
    def __init__(self,row,col):
        self.row = row
        self.col = col
        self.x = row * NODE_WIDTH
        self.y = col * NODE_WIDTH
        self.color = WHITE
        self.neighbours = []
    
    def get_pos(self):
        """
        Return the row and column coordinates of node
        """
        return self.row, self.col
    
    def reset(self):
        """
        Reset the node
        """
        self.color = WHITE

    def close_node(self):
        """
        Close node when it is removed from open set and no longer being checked
        """
        self.color = RED
    
    def open_node(self):
        """
        Open node when it is added to the open set and joins the queue to be checked
        """
        self.color = GREEN
    
    def make_barrier(self):
        """
        Make node a barrier that paths can not travel through
        """
        self.color = BLACK
    
    def make_end(self):
        """
        Make node the path destination
        """
        self.color = BLUE
    
    def make_start(self):
        """
        Make node the path origin node
        """
        self.color = ORANGE

    def make_path(self):
        """
        Highlight node purple if used in shortest path 
        """
        self.color = PURPLE

    def draw(self, win):
        """
        Draw node in it's given colour at given coordinates
        """
        pg.draw.rect(win, self.color, (self.x, self.y, NODE_WIDTH, NODE_WIDTH))

    def update_neighbours(self, grid):
        """
        Give node a list of neighbours based on the available nodes surrounding it
        """
        if self.row < ROWS - 1 and not grid[self.row + 1][self.col].color == BLACK:
            self.neighbours.append(grid[self.row +  1][self.col])
        
        if self.row > 0 and not grid[self.row - 1][self.col].color == BLACK:
            self.neighbours.append(grid[self.row - 1][self.col])
        
        if self.col > 0 and not grid[self.row][self.col-1].color == BLACK:
            self.neighbours.append(grid[self.row][self.col-1])
        
        if self.col < ROWS - 1 and not grid[self.row][self.col+1].color == BLACK:
            self.neighbours.append(grid[self.row][self.col+1])

    def __lt__(self,other):          
        """
        Dunder method that allows use of operators < > on instances of class
        """
        return False


def get_dist(node1, node2):
    """
    Return Manhatan distance between nodes
    """
    x1, y1 = node1.get_pos()
    x2, y2 = node2.get_pos()

    return abs(x1 - x2) + abs(y1 - y2)


def update_table(df, current_node, g_score, h_score, came_from):
    """
    Update dataframe with new scores for chosen node
    """
    for input in [[g_score,'g_score'], [g_score + h_score, 'f_score'], [h_score, 'h_score'], [came_from, 'came_from']]:
        df.loc[df['node']==current_node, input[1]] = input[0]

    return df


def algorithm(grid, start, end):
    """
    A star algorithm implementation

    g_score: distance from start node to current node
    h_score: distance from current node to end node ignoring paths
    f_score: h_score + g_score
    came_from: The node that the current node came from to achieve its g_score

    Algorithm starts with start node
    The current nodes neighbours are then checked if they have already been checked
    If they haven't, they are added to the open set which is the queue for checking
    If they have been checked, their g_score is compared to the g_score already in the table
    If it is less than the exisiting one, the came_from and g_score in the table are updated 
    The next node to be checked is the node with the lowest f_score
    If the open_set is ever empty, it means there is no path
    Continues until end node is found
    """
    open_set = []
    open_set_hash = []
    open_set.append((0,start))
    open_set_hash.append(start)
    df = pd.DataFrame(columns = ['node','g_score','h_score','f_score','came_from'])
    g_score = float('inf')
    f_score = float('inf')
    h_score = float('inf')
    i = 0

    for row in grid:
        for node in row:
            df.loc[len(df)] = [node, g_score, h_score, f_score, 0]  # Initialise scores as defaults for each node
    
    # Start algorithm
    current_node = start
    g_score_prev = 0
    g_score = 0
    h_score = get_dist(start,end)
    came_from = None

    df = update_table(df, current_node, g_score, h_score, came_from)

    while min(open_set)[1] != end:
        # Get the next nodes to check (neighbours of current)
        current_node = min(open_set)[1]
        open_set.remove(min(open_set))

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()

        if current_node == end:
            node = end
            print(df)
            
            while node != start:
                node.make_path()
                node = df[df['node']==node]['came_from'].reset_index(drop=True)[0]
            
            return 1

        next_nodes = current_node.neighbours
        min_f = float('inf')
        current_g = df.loc[df['node']==current_node, 'g_score'].reset_index(drop=True)[0]

        # Iterate through the neighbours and find which has best f score, then that node becomes the best

        for node in next_nodes:

            # if node != start and node != end:
            #     node.open_node()

            h_score = get_dist(node, end)
            g_score = current_g + 1
            f_score = h_score + g_score

            if g_score <= df.loc[df['node']==node, 'g_score'].reset_index(drop=True)[0]:    # Makes sure only the node with best g_score is in table
                df = update_table(df, node, g_score, h_score, current_node)
                if node not in open_set_hash and node != end:
                    open_set.append((f_score, node))
                    open_set_hash.append(node)  # Make sure nodes don't get checked multiple times as this will greatly slow the code down 
                    node.open_node()  

            elif node != start and node != end:
                node.close_node()
            
            if node == end:
                path_node = df[df['node']==end]['came_from'].reset_index(drop=True)[0]
                print('finished')
                print(df)
                while path_node != start:
                    path_node.make_path()
                    path_node = df[df['node']==path_node]['came_from'].reset_index(drop=True)[0]

                return 1
                
        draw(WIN,grid)

        if current_node != start:
            current_node.close_node()
        
        if len(open_set) == 0:
            print('no route')
            return 1

    return False

      
def make_grid(rows, width):
    """
    Make a list of lists representing the nodes in each row
    """
    grid = []
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i,j)
            grid[i].append(node)
    
    return grid

def draw(win, grid):
    """
    Update the display to show any changes
    """
    win.fill(WHITE)
    for row in grid:
        for node in row:
            node.draw(win)

    pg.display.update()



def get_clicked_pos(pos):
    """
    Return the row and column that has been clicked
    """
    y,x = pos
    row = y//NODE_WIDTH
    col = x//NODE_WIDTH
    
    return row, col

def title_screen(win, grid):
    """
    Draw title screen showing instructions
    """
    draw(win, grid)
    win.blit(TITLE_IMAGE,(10,10))
    pg.display.update()
    title = 1
    while title == 1:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                title = 0
                break


def main(win):
    """
    Calls the above functions
    """
    grid = make_grid(ROWS, WIDTH)

    start = None
    end = None

    run = True
    started = False

    title_screen(win, grid)

    while run:
        draw(win, grid)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()

            if not started:
                if pg.mouse.get_pressed()[0]:
                    pos = pg.mouse.get_pos()
                    row, col = get_clicked_pos(pos)
                    if int(row) >= ROWS:    # Ensures row and col are in grid
                        row = ROWS - 1
                    if int(col) >= ROWS:
                        col = ROWS - 1
                    if int(col) < 0:
                        col = 0
                    if int(row) < 0:
                        row = 0

                    node = grid[int(row)][int(col)]
                    if not start:
                        start = node
                        start.make_start()
                    
                    elif not end and node != start:
                        end  = node
                        end.make_end()
                    
                    elif node != end and node != start:
                        node.make_barrier()

                elif pg.mouse.get_pressed()[2]:
                    pos = pg.mouse.get_pos()
                    row, col = get_clicked_pos(pos)
                    node = grid[int(row)][int(col)]
                    node.reset()
                    if node == start:
                        start = None
                    elif node == end:
                        end = None
                
            if event.type == pg.KEYDOWN and event.key == pg.K_SPACE and not started:
                started = 1
                for row in grid:
                    for node in row:
                        node.update_neighbours(grid)
                print('started')
                algorithm(grid, start, end)
            
            if event.type == pg.KEYDOWN and event.key == pg.K_r:
                for row in grid:
                    for node in row:
                        node.reset()    # Reset all nodes in grid
                        node.neighbours = []

                started = 0         
                start = None
                end = None          # ready to populate again

main(WIN)

