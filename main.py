from typing import List, Tuple
#from local_driver import Alg3D, Board # ローカル検証用
# from framework import Alg3D, Board # 本番用
import math

class MyAI():
    def __init__(self):
        # all possible winning lines
        self.lines = self.generate_lines()
        # check if the game is over
        self.over = False
        
    def get_move(
        self,
        board: List[List[List[int]]], # 盤面情報
        player: int, # 先手(黒):1 後手(白):2
        last_move: Tuple[int, int, int] # 直前に置かれた場所(x, y, z)
    ) -> Tuple[int, int]:
        # ここにアルゴリズムを書く
        self.player = player
        # HERE OPTIMISE
        self.alpha_beta_minimax(board, True, 0, 2, 0, 0)
        return (0, 0)

#     fonction MINIMAX( p) est


    def result(self, board, action):
        """
            return the board that result from a move
            board: current board
            move: (x,y,z) where to play
            player: which player is playing
            return new board
        """ 
        board[action[0]][action[1]][action[2]] = self.player
        return board

    def generate_lines(self):
        lines = []
        rng = range(4)
        for z in rng:
            for y in rng:
                lines.append([(x,y,z) for x in rng])
        for z in rng:
            for x in rng:
                lines.append([(x,y,z) for y in rng])
        for y in rng:
            for x in rng:
                lines.append([(x,y,z) for z in rng])

        for z in rng:
            lines.append([(i,i,z) for i in rng])
            lines.append([(i,3-i,z) for i in rng])
        for y in rng:
            lines.append([(i,y,i) for i in rng])
            lines.append([(i,y,3-i) for i in rng])
        for x in rng:
            lines.append([(x,i,i) for i in rng])
            lines.append([(x,i,3-i) for i in rng])

        # diagonal
        lines.append([(i,i,i) for i in rng])
        lines.append([(i,i,3-i) for i in rng])
        lines.append([(i,3-i,i) for i in rng])
        lines.append([(3-i,i,i) for i in rng])
        return lines

    def is_terminal(self, board):
        """
            check if the game ended
            return 1 if ai win -1 if lose and 0 equal
        """
        enemy = 1 if self.player == 2 else 2
        for line in self.lines:
            if all(board[z][y][x] == self.player for (x,y,z) in line):
                self.over = True
                self.end_value = 1
                break
            elif all(board[z][y][x] == enemy for (x,y,z) in line):
                self.over = True
                self.end_value = -1
                break
        # if board is full
        if all(board[3][y][x] != 0 for x in range(4) for y in range(4)):
            self.over = True
            self.end_value = 0
        return False


    def evaluate(self, board, lines):
        """
            END CONDITION
            return 100 if ai win -100 if lose and 0 equal
        """
		# score = 0
        enemy = 1 if self.player == 2 else 2

        if self.over:
            return self.end_value * 100
        
		# Heuristic scoring
        for line in lines:
			# Example line : [(0,0,0), (1,1,1), (2,2,2), (3,3,3)]
			# Example values : [-1, 1, 0, 2]
            values = [board[x,y,z] for (x,y,z) in line]
			
            if values.count(self.player) == 3 and values.count(0) == 1:
                score += 10
            elif values.count(self.player) == 2 and values.count(0) == 2:
                score += 1

			if values.count(enemy) == 3 and values.count(0) == 1:
				score -= 10
			elif values.count(enemy) == 2 and values.count(0) == 2:
				score -= 1
		
        return score

    def legal_move(self, board):
        """
            use to determine how many legal moves are possible
        """
        action_arr = []

        for plane_i in range(board.length):
            for row_i in range(plane_i):
                for space_i in range(row_i):
                    if board[plane_i][row_i][space_i] > 0 \
                        and (board.length - 1 == plane_i \
                        or board[plane_i + 1][row_i][space_i] == 0 ):

                        action_arr.append(tuple(plane_i, row_i, space_i))
        return action_arr

    def alpha_beta_minimax(self, board, isMaximiser, depth, max_depth, alpha, beta):
        """
            isMaximiser: is the computer turn to check in the three
            depth: how far in the three you are
            max_deth: maximmum depth
        """
        if depth == 0 or self.is_terminal(board) or depth == max_depth:
            return self.evaluate(board)

        if isMaximiser:
            max_eval = -math.inf
            for action in self.legal_move(board):
                eval = self.minimax(self.result(board, action), False, depth - 1, max_depth, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for action in self.legal_move(board):
                eval = self.alpha_beta_minimax(self.result(board, action), True, depth - 1, max_depth, alpha, beta)
                min_eval = max(max_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval


# TEST
def main():
    # Create a 4x4x4 cube filled with 0
    cube = [[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    
    ai = MyAI()
    result = ai.is_terminal(cube, 1)
    print("Result of is_terminal on empty board:", result)  # Expected output:
if __name__ == "__main__":
