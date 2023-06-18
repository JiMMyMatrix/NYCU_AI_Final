import random
import time
import math

SIZE = 3
EXPLORATION_WEIGHT = 2**0.5
ITERATION_TIME = 2500

class MCTS:
    def __init__(self):
        self.child_dict = dict()

    def run(self, node):
        path = self.select(node)
        leaf = path[-1]
        tmp = self.expand(leaf)
        leaf = tmp if tmp != None else leaf
        # print(f'leaf:\n{leaf}\n')
        reward = self.simulate(leaf)
        self.backpropagate(path, reward)

    def select(self, node):
        path = []
        i=0
        while True:
            path.append(node)
            if node.children == None:
                return path
            else:
                node = self.ucb_select(node)
            i+=1
    def expand(self, node):
        game_end, dead_end = check_game_end(node.state, False)
        if game_end or dead_end:
            return None
        
        if node.children == None and not node.visited:
            node.children = node.find_valid_move()
            for child in node.children:
                self.child_dict[child] = node
            node.visited = True
            if node.children == None:
                return None
            else:
                return self.ucb_select(node)
        else:
            return None
        
    def simulate(self, node):
        end = False
        reward = None
        i=0
        while True:
            if end is True:
                return reward
            end, reward, node = node.find_random_move()
            i+=1

    def backpropagate(self, path, reward):
        for node in reversed(path):
            node.N += 1    
            node.T += reward

    def ucb_select(self, node):
        def ucb(n):
            return math.inf if n.N == 0 else n.T/n.N + EXPLORATION_WEIGHT * math.sqrt(
                    math.log(self.child_dict[n].N)/n.N
            )
        
        return max(node.children, key = ucb)


class MCTSNode():
    def __init__(self, state = None, turn = False, visited = False, rc = None, sub = None):
        self.state = state
        self.T = 0
        self.N = 0
        self.turn = turn
        self.visited = visited
        self.rc = rc
        self.sub = sub
        self.children = None

    def find_valid_move(self):
        children_list = []
        for sub in range(1, 4):
            for row_or_col in range(6):
                sel_node = self.new_children_node(sub, row_or_col)
                if sel_node != None:
                    children_list.append(sel_node)
        if len(children_list) == 0:
            return None
        else:
            return children_list

    def find_random_move(self):
        tmp = self.state
        game_end, dead_end = check_game_end(tmp, dead_end=False)
        if dead_end:
            if self.turn == True:
                return True, 0, None
            else:
                return True, 6, None
        elif game_end and not dead_end:
            if self.turn == True:
                return True, 8, None
            elif self.sub == 1:
                return True, 5, None
            elif self.sub == 2:
                return True, 2, None
            elif self.sub == 3:
                return True, 1, None
            else:
                return True, 0, None
        else:
            return False, None, random.choice(self.find_valid_move())
    
    def new_children_node(self, sub, row_or_col):
        tmp = self.state
        board = [[0 for j in range(SIZE)] for i in range(SIZE)]
        # print(f'tmp: {tmp[0]}')
        
        if check_valid(tmp, row_or_col, sub):
            if row_or_col > 2:
                for i in range(SIZE):
                    for j in range(SIZE):
                        if row_or_col-3-j == 0:
                            board[i][j] = tmp[i][j]-sub
                        else:
                            board[i][j] = tmp[i][j]
            else:
                for i in range(SIZE):
                    for j in range(SIZE):
                        if row_or_col-i == 0:
                            board[i][j] = tmp[i][j]-sub
                        else:
                            board[i][j] = tmp[i][j]
            turn = not self.turn
            visited = False
            return MCTSNode(board, turn, visited, row_or_col, sub)
        
        return None
    
    def __repr__(self):
        return f'state = {self.state},\nturn = {self.turn}, visited = {self.visited}\n'

    def __iter__(self):
        return self

def board_min(board):
    min = board[0][0]
    for i in range(SIZE):
        for j in range(SIZE):
            if board[i][j] <= min:
                min = board[i][j]
    
    return min

def random_init(board):
    random.seed()
    for i in range(SIZE):
        for j in range(SIZE):
            board[i][j] = random.randint(50, 99)

def given_init():
    given = [[19, 0, 0], [13, 11, 5], [0, 17, 21]]
    
    return given

def print_board(board):
    for i in range(SIZE):
        for j in range(SIZE):
            print(board[i][j], end=" ")
        print()

def check_valid(board, row_or_col, subtract):
    min_val = float('inf')

    if subtract <= 0 or subtract > 3: # subtract not 1, 2 or 3
        return False
    
    if row_or_col >= 0 and row_or_col < SIZE: # row 
        for i in range(SIZE): 
            min_val = min(board[row_or_col][i], min_val) 
        if min_val == 0: 
            return False 
        elif min_val == 1 and subtract > 1:
            return False
        elif min_val == 2 and subtract > 2:
            return False
        return True
    elif row_or_col >= SIZE and row_or_col < SIZE*2: # col 
        row_or_col -= SIZE 
        for i in range(SIZE): 
            min_val = min(board[i][row_or_col], min_val) 
        if min_val == 0: 
            return False 
        elif min_val == 1 and subtract > 1:
            return False
        elif min_val == 2 and subtract > 2:
            return False
        return True
    else:
        return False

def board_subtract(board, row_or_col, subtract):
    if row_or_col >= 0 and row_or_col < SIZE: # row 
        for i in range(SIZE): 
            board[row_or_col][i] -= subtract 
    elif row_or_col >= SIZE and row_or_col < SIZE*2: # col
        row_or_col -= SIZE
        for i in range(SIZE):
            board[i][row_or_col] -= subtract

def check_diagonal(board):
    diagonal1 = True
    diagonal2 = True
    for i in range(SIZE):
        diagonal1 &= (board[i][i] == 0)
        diagonal2 &= (board[i][SIZE - i - 1] == 0)
    return diagonal1 or diagonal2

def check_row(board, row): # check num in row are all 0
    for i in range(SIZE):
        if board[row][i] != 0:
            return False
    return True

def check_col(board, col):
    for i in range(SIZE):
        if board[i][col] != 0:
            return False
    return True

def check_game_end(board, dead_end):
    game_end = False
    row_end = True
    col_end = True

    # end condition 1 : all the numbers in any row, column, or diagonal become 0
    for i in range(SIZE):
        game_end |= check_row(board, i)
        game_end |= check_col(board, i)
    game_end |= check_diagonal(board)

    # end condition 2 : every row or column contains the number 0
    if not game_end:
        for i in range(SIZE):
            row_exist_zero = False
            col_exist_zero = False
            for j in range(SIZE):
                if board[i][j] == 0:
                    row_exist_zero = True
                if board[j][i] == 0:
                    col_exist_zero = True
            if not row_exist_zero:
                row_end = False
            if not col_exist_zero:
                col_end = False
        dead_end = row_end and col_end
        game_end |= dead_end

    return game_end, dead_end

def make_your_move(board):
    state = [[0 for j in range(SIZE)] for i in range(SIZE)]
    for i in range(SIZE):
        for j in range(SIZE):
            state[i][j] = board[i][j]

    current_board = MCTSNode(state, turn = False, visited = False, rc = None, sub = None)
    tree = MCTS()
    for i in range(ITERATION_TIME):
        # output_file.write(f'Make your move {i}\n')
        tree.run(current_board)

    max_sub1 = 0
    max_sub2 = 0
    max_sub3 = 0
    max = 0
    tmp1 = None
    tmp2 = None
    tmp3 = None
    choose = None

    if board_min(current_board.state) >= 16:
        for child in tree.child_dict.keys():
            if tree.child_dict[child] == current_board:
                if child.T/child.N >= max_sub1 and child.sub == 1:
                    tmp1 = child
                    max_sub1 = child.T/child.N
                if child.T/child.N >= max_sub2 and child.sub == 2:
                    tmp2 = child
                    max_sub2 = child.T/child.N
                if child.T/child.N >= max_sub3 and child.sub == 3:
                    tmp3 = child
                    max_sub3 = child.T/child.N
                
        if tmp1:
            choose = tmp1
        elif tmp2:
            choose = tmp2
        else:
            choose = tmp3
    else:
        for child in tree.child_dict.keys():
            if tree.child_dict[child] == current_board:
                if child.T/child.N >= max:
                    max = child.T/child.N
                    choose = child
                    
    row_or_col = choose.rc
    subtract = choose.sub

    # arr = [[11, 0, 0],
    #        [0, 5, 6],
    #        [1, 0, 6]]
    
    # print(f'\n{check_game_end(arr, False)}\n')


    return row_or_col, subtract

def opponent_move(board):

    #row_or_col = int(input())
    #subtract = int(input())

    valid = False
    while(not valid):
        row_or_col = random.randint(0, 5)
        subtract = random.randint(1, 3)
        valid = check_valid(board, row_or_col, subtract)

    return row_or_col, subtract

if __name__ == "__main__":

    board = [[0 for i in range(SIZE)] for j in range(SIZE)]
    player = 0 # player 0 goes first
    total_cost = [0, 0] # total cost for each player
    your_turn = True
    game_end = False
    dead_end = False
    reward = 15
    penalty = 7

    print("Board initialization…\n")

    random_init(board)   # initialize board with positive integers
    # board = given_init() # initialize board with given intergers (for testing only)

    while not game_end:
        print("Current board:")
        print_board(board)
        print("\nPlayer", player, "'s turn:\n")
        # input("Press any key to continue…")
        row_or_col = 0  # row1, row2, row3 -> 0, 1, 2 ; col1, col2, col3 -> 3, 4, 5
        subtract = 0  # number to subtract

        if your_turn:
            start = time.time()
            row_or_col, subtract = make_your_move(board)
            end = time.time()

            print("Time:", end - start, "seconds\n")
        else:
            #start = time.time()
            row_or_col, subtract = opponent_move(board)
            #end = time.time()

            #print("Time:", end - start, "seconds\n")

        print("Player", player, "'s move: row_or_col:", row_or_col, "subtract:", subtract, "\n")

        print("Valid checking...", end="")
        valid = check_valid(board, row_or_col, subtract)  # legal move checking
        if valid: print("The move is vaild.")
        else: print("The move is invalid, game over.")
        # input("Press any key to continue…")

        board_copy = [row[:] for row in board]  # make a copy of the board
        board_subtract(board_copy, row_or_col, subtract)

        # update the board
        board = board_copy

        # update player's total cost
        total_cost[player] += subtract

        print("Player 0 total cost:", total_cost[0])
        print("Player 1 total cost:", total_cost[1])
        print("--------------------------------------\n")

        # check if game has ended
        game_end, dead_end= check_game_end(board, dead_end)

        if not game_end:
            # switch to other player
            your_turn = not your_turn
            player = (player + 1) % 2

    print("Final board:")
    print_board(board)
    if not dead_end:
        print("Player", player, "ends with a diagonal/row/col of 0's!")
        total_cost[player] -= reward
    else:
        print("Player", player, "ends with a dead end!")
        total_cost[player] += penalty

    print("Player 0 total cost:", total_cost[0])
    print("Player 1 total cost:", total_cost[1])
    # input("Press Enter to exit…")