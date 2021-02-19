import math
import agent

###########################
# Alpha-Beta Search Agent #
###########################

class AlphaBetaAgent(agent.Agent):
    """Agent that uses alpha-beta search"""

    # Class constructor.
    #
    # PARAM [string] name:      the name of this player
    # PARAM [int]    max_depth: the maximum search depth
    def __init__(self, name, max_depth):
        super().__init__(name)
        # Max search depth
        self.max_depth = max_depth

    # Pick a column.
    #
    # PARAM [board.Board] brd: the current board state
    # RETURN [int]: the column where the token must be added
    #
    # NOTE: make sure the column is legal, or you'll lose the game.
    def go(self, brd):
        """Search for the best move (choice of column for the token)"""
        self.player = brd.player
        return self.alpha_beta_search(self.max_depth, -10000000, 100000000, True, brd)[1]

    def alpha_beta_search(self, depth, alpha, beta, is_maximizing, current_board):
        children = self.get_successors(current_board)
        if current_board.get_outcome() != 0 or depth == 0 or len(children) == 0:
            return self.heuristic(current_board, depth), -1
        best_column = 999
        if is_maximizing:
            value = -1000000000000
            for child in children:
                new_val = self.alpha_beta_search(depth - 1, alpha, beta, False, child[0])[0]
                if new_val >= value:
                    best_column = child[1]
                    value = new_val
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:  # minimizing
            value = 1000000000000
            for child in children:
                new_value = self.alpha_beta_search(depth - 1, alpha, beta, True, child[0])[0]
                value = min(value, new_value)
                beta = min(beta, value)
                if beta <= alpha:
                    break
        return value, best_column

    def heuristic(self, brd, depth_remaining):
        current_depth = self.max_depth - depth_remaining
        if self.player == 1:
            other_player = 2
        else:
            other_player = 1
        if brd.get_outcome() == self.player:
            return 100000000 - 100 * current_depth
        if brd.get_outcome() == other_player:
            return -100000000 + 100 * current_depth
        connected_lines = self.count_usable_connected_in_board(brd)
        score = 0
        for line in connected_lines[self.player]:
            score += (10 ** (line - 1)) * connected_lines[self.player][line]
        for line in connected_lines[other_player]:
            score -= (10 ** (line - 1)) * connected_lines[other_player][line]
        return score

    def count_usable_connected_in_board(self, board):
        num_lines_of_length_n = {
            1: {},
            2: {}
        }
        for i in range(board.n + 1):
            num_lines_of_length_n[1][i] = 0
            num_lines_of_length_n[2][i] = 0
        visited_coords = []
        for y in range(board.h):
            for x in range(board.w):
                if board.board[y][x] == 0 or (x, y) in visited_coords:
                    continue
                player = board.board[y][x]
                lines, visited = self.get_usable_line_length_in_every_dir(board, x, y)
                visited_coords.extend(visited)
                for line in lines:
                    num_lines_of_length_n[player][line] += 1
        return num_lines_of_length_n

    def get_usable_line_length_in_every_dir(self, board, x, y):
        lines = []
        dirs = [
            (0, 1),
            (1, 0),
            (1, 1),
            (-1, 1)
        ]
        visited_coords = []
        for direction in dirs:
            line, visited = self.get_usable_line_length(board, x, y, direction[0], direction[1])
            visited_coords.extend(visited)
            lines.append(line)
        return lines, visited_coords

    def get_usable_line_length(self, brd, x, y, dir_x, dir_y):
        line_len = 0
        player = brd.board[y][x]
        x1, x2 = x, x
        y1, y2 = y, y
        visited_coords = []
        while self.within_board(brd, x1, y1) and brd.board[y1][x1] == player:
            visited_coords.append((x1, y1))
            line_len += 1
            x1 += dir_x
            y1 += dir_y
        while self.within_board(brd, x2, y2) and brd.board[y2][x2] == player:
            visited_coords.append((x2, y2))
            line_len += 1
            x2 -= dir_x
            y2 -= dir_y
        if (not self.within_board(brd, x1, y1) or brd.board[y1][x1] != 0) and \
                (not self.within_board(brd, x2, y2) or brd.board[y2][x2] != 0):
            return 0, []
        return line_len, visited_coords

    # Checks if the given coordinate is within the board
    def within_board(self, brd, x, y):
        return not(y < 0 or y >= brd.h or x < 0 or x >= brd.w)

    # Get the successors of the given board.
    #
    # PARAM [board.Board] brd: the board state
    # RETURN [list of (board.Board, int)]: a list of the successor boards,
    #                                      along with the column where the last
    #                                      token was added in it
    def get_successors(self, brd):
        """Returns the reachable boards from the given board brd. The return value is a tuple (new board state, column number where last token was added)."""
        # Get possible actions
        freecols = brd.free_cols()
        # Are there legal actions left?
        if not freecols:
            return []
        # Make a list of the new boards along with the corresponding actions
        succ = []
        for col in freecols:
            # Clone the original board
            nb = brd.copy()
            # Add a token to the new board
            # (This internally changes nb.player, check the method definition!)
            nb.add_token(col)
            # Add board to list of successors
            succ.append((nb,col))
        return succ
