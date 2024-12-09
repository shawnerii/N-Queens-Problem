import time
import sys
from multiprocessing import Pool, cpu_count

def backtrack(n, row, cols, majors, minors):
    """
    Recursive backtracking function using bitmasking.
    
    Args:
        n (int): Size of the board (N x N).
        row (int): Current row to place a queen.
        cols (int): Bitmask of occupied columns.
        majors (int): Bitmask of occupied major diagonals.
        minors (int): Bitmask of occupied minor diagonals.
    
    Returns:
        int: Number of solutions found from this state.
    """
    if row == n:
        return 1
    available_positions = ~(cols | majors | minors) & ((1 << n) - 1)
    count = 0
    while available_positions:
        position = available_positions & -available_positions  # Rightmost 1
        available_positions -= position
        count += backtrack(
            n,
            row + 1,
            cols | position,
            (majors | position) << 1,
            (minors | position) >> 1
        )
    return count

def worker(args):
    """
    Worker function to count solutions with the first queen placed in the given column.
    
    Args:
        args (tuple): A tuple containing:
                      - n (int): Size of the board.
                      - col (int): Column index for the first queen.
    
    Returns:
        int: Number of solutions for this placement.
    """
    n, col = args
    position = 1 << col
    return backtrack(n, 1, position, position << 1, position >> 1)

def count_n_queens_parallel(n):
    """
    Counts all distinct solutions to the N-Queens problem using bitmasking, symmetry breaking, and multiprocessing.
    
    Args:
        n (int): Size of the board (N x N).
    
    Returns:
        int: Total number of distinct solutions.
    """
    total_solutions = 0
    half = n // 2  # Symmetry breaking: Only iterate through half the columns
    
    # Prepare arguments for worker processes
    args = [(n, col) for col in range(half)]
    
    # Determine the number of available CPU cores
    num_processes = cpu_count()
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(worker, args)
        total_solutions += sum(results) * 2  # Mirror the solutions
    
    # If N is odd, handle the central column separately
    if n % 2:
        central_col = n // 2
        central_args = (n, central_col)
        central_count = backtrack(n, 1, 1 << central_col, (1 << central_col) << 1, (1 << central_col) >> 1)
        total_solutions += central_count
    
    return total_solutions

def main():
    """
    Main function to execute the N-Queens solver for specified test cases.
    """
    test_cases = [4, 5, 6, 7, 8, 10, 12, 14, 16]  # Modify as needed
    
    # Increase recursion limit to handle deep recursion for larger N
    sys.setrecursionlimit(10000)
    
    for n in test_cases:
        start_time = time.time()
        total_solutions = count_n_queens_parallel(n)
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"N={n}: Total Solutions = {total_solutions}, Execution Time = {execution_time:.4f} seconds")

if __name__ == "__main__":
    main()