from ortools.sat.python import cp_model
import numpy as np
import collections


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Show intermediate solutions."""

    def __init__(self, board, variables, edge):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__edge = edge
        self.__solution_count = 0
        self.__board = board

    def OnSolutionCallback(self):
        
        try:
            self.__solution_count += 1
            solved = []
            direction = []
            for v in self.__variables:
                solved.append([self.Value(x) for x in v])
                                
            for v in self.__edge:
                color_x_y_x_y = v.split(' ')[1:]
                color, y_prev, x_prev, y_cur, x_cur = color_x_y_x_y
                if (x_prev != x_cur or y_prev != y_cur) and self.Value(self.__edge[v]):
                    direction.append([color, x_prev, y_prev, x_cur, y_cur])
            plot_directed_solution_with_dir(self.__board, np.array(solved), direction)

        except Exception as e:
            print(e)
            traceback.print_exc()
            raise e
        

    def solution_count(self):
        return self.__solution_count


def plot_directed_solution_with_dir(board, S, direction):
    import matplotlib.pyplot as plt
    
    ax = plt.gca()
    colors = plt.cm.tab10.colors
    M, N = np.array(S).shape[0], np.array(S).shape[1]
    
    for item in direction:
        prev_x, prev_y = int(item[1]), int(item[2])
        cur_x, cur_y = int(item[3]),  int(item[4])
        ax.plot([prev_x, cur_x], [prev_y, cur_y], color=colors[int(item[0])], lw=15)
        
    for i in range(M):
        for j in range(N):
            if board[i][j] != 0:
                ax.scatter(j, i, s=500, color=colors[board[i][j]] if board[i][j]>0 else "black" )
                            
                            
    ax.set_ylim(M - 0.5, -0.5)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_yticks([i + 0.5 for i in range(M - 1)], minor=True)
    ax.set_xticks([j + 0.5 for j in range(N - 1)], minor=True)
    ax.grid(b=True, which='minor', color='white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='both', which='both', length=0)
    plt.show()

def validate_board_and_count_colors(board):
    assert isinstance(board, list)
    assert all(isinstance(row, list) for row in board)
    assert len(set(map(len, board))) == 1
    colors = collections.Counter(square for row in board for square in row)
    del colors[0]
    neg_counter = 0
    for color in colors:
        if color < 0:
            neg_counter += 1
    for i in range(neg_counter+1):
        del colors[-i]
    assert all(count == 2 for count in colors.values())
    num_colors = len(colors)
    assert set(colors.keys()) == set(range(1, num_colors + 1))
    return num_colors


def main(board):
    num_colors = validate_board_and_count_colors(board)
    model = cp_model.CpModel()
    solution = [
        [square or model.NewIntVar(1, num_colors, "") for (j, square) in enumerate(row)]
        for (i, row) in enumerate(board)
    ]
    edge = {}
    true = model.NewBoolVar("")
    model.AddBoolOr([true])
    for color in range(1, num_colors + 1):
        endpoints = []
        arcs = []
        for i, row in enumerate(board):
            for j, square in enumerate(row):
                if square == color:
                    endpoints.append((i, j))
                else:
                    arcs.append(((i, j), (i, j)))
                if i < len(board) - 1:
                    arcs.append(((i, j), (i + 1, j)))
                if j < len(row) - 1:
                    arcs.append(((i, j), (i, j + 1)))
        (i1, j1), (i2, j2) = endpoints
        k1 = i1 * len(row) + j1
        k2 = i2 * len(row) + j2
        arc_variables = [(k2, k1, true)]
        for (i1, j1), (i2, j2) in arcs:
            k1 = i1 * len(row) + j1
            k2 = i2 * len(row) + j2
            edge_name = f"color_i1_j1_i2_j2 {color} {i1} {j1} {i2} {j2}"
            edge[edge_name] = model.NewBoolVar(edge_name)
            if k1 == k2:
                model.Add(solution[i1][j1] != color).OnlyEnforceIf(edge[edge_name])
                arc_variables.append((k1, k1, edge[edge_name]))
            else:
                model.Add(solution[i1][j1] == color).OnlyEnforceIf(edge[edge_name])
                model.Add(solution[i2][j2] == color).OnlyEnforceIf(edge[edge_name])
                forward = model.NewBoolVar("")
                backward = model.NewBoolVar("")
                model.AddBoolOr([edge[edge_name], forward.Not()])
                model.AddBoolOr([edge[edge_name], backward.Not()])
                model.AddBoolOr([edge[edge_name].Not(), forward, backward])
                model.AddBoolOr([forward.Not(), backward.Not()])
                arc_variables.append((k1, k2, forward))
                arc_variables.append((k2, k1, backward))
        model.AddCircuit(arc_variables)
        
    #### print multiple solution
    solver = cp_model.CpSolver()
    solution_printer = VarArraySolutionPrinter(board,solution, edge)
    status = solver.SearchForAllSolutions(model, solution_printer)
    print()
    print('Solutions found : %i' % solution_printer.solution_count())


if __name__ == "__main__":
    main(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 3, 3],
            [0, 2, 2, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            
        ]
    )
