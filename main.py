from typing import List, Tuple
#from local_driver import Alg3D, Board # ローカル検証用
from framework import Alg3D, Board # 本番用

class MyAI(Alg3D):
    def get_move(
        self,
        board: List[List[List[int]]], # 盤面情報
        player: int, # 先手(黒):1 後手(白):2
        last_move: Tuple[int, int, int] # 直前に置かれた場所(x, y, z)
    ) -> Tuple[int, int]:
        # ここにアルゴリズムを書く

        return (0, 0)

    def minimax(self, board: List[List[List[int]]], player: int, depth: int) -> Tuple[int, int]:
        if depth == 0:
            return self.evaluate(board, player)
        best_score = float('-inf')
        best_move = None
        for move in self.get_moves(board, player):
            new_board = self.make_move(board, move, player)
            score = self.minimax(new_board, player, depth - 1)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def evaluate(self, board: List[List[List[int]]], player: int) -> int:
        return 0

    def get_moves(self, board: List[List[List[int]]], player: int) -> List[Tuple[int, int]]:
        return []

    def make_move(self, board: List[List[List[int]]], move: Tuple[int, int], player: int) -> List[List[List[int]]]:
        return board
