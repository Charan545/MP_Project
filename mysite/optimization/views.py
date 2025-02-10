# optimization/views.py
from django.shortcuts import render
from .forms import OptimizationForm
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import linprog

def home(request):
    return render(request, 'optimization/home.html')

def index(request):
    form = OptimizationForm()
    return render(request, 'optimization/index.html', {'form': form})

def solve(request):
    if request.method == 'POST':
        form = OptimizationForm(request.POST)
        if form.is_valid():
            obj_func = form.cleaned_data['objective_function']
            constraints = form.cleaned_data['constraints']
            method = form.cleaned_data['method']

            if method == 'simplex':
                solution = solve_simplex(obj_func, constraints)
            elif method == 'graphical':
                solution = solve_graphical(obj_func, constraints)
            elif method == 'modi':
                solution = solve_modi(obj_func, constraints)
            else:
                solution = "Invalid method selected."
            
            return render(request, 'optimization/result.html', {'solution': solution})
    else:
        form = OptimizationForm()
    return render(request, 'optimization/index.html', {'form': form})

def solve_simplex(obj_func, constraints):
    try:
        c = np.array([float(x) for x in obj_func.strip().split()])
        lhs_ineq = []
        rhs_ineq = []
        for constraint in constraints.strip().split("\n"):
            parts = constraint.strip().split("<=")
            if len(parts) != 2:
                return "Error: Invalid constraint format."
            lhs_ineq.append([float(x) for x in parts[0].strip().split()])
            rhs_ineq.append(float(parts[1].strip()))
        A_ub = np.array(lhs_ineq)
        b_ub = np.array(rhs_ineq)
        res = linprog(-c, A_ub=A_ub, b_ub=b_ub, method='simplex')
        return f'Optimal Solution: {res.x}, Objective Value: {-res.fun}'
    except Exception as e:
        return f'Error: {e}'

def solve_graphical(obj_func, constraints):
    try:
        plt.figure()
        for constraint in constraints.strip().split("\n"):
            parts = constraint.strip().split("<=")
            if len(parts) != 2:
                return "Error: Invalid constraint format."
            coef = [float(x) for x in parts[0].strip().split()]
            b = float(parts[1].strip())
            x_vals = np.linspace(0, 10, 100)
            y_vals = (b - coef[0] * x_vals) / coef[1]
            plt.plot(x_vals, y_vals, label=constraint)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        plt.title("Graphical Method Solution")
        plt.grid()
        static_dir = os.path.join(os.getcwd(), "static")
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
        graph_path = os.path.join(static_dir, "graph.png")
        plt.savefig(graph_path)
        plt.close()
        return '<img src="/static/graph.png" alt="Graphical Solution">'
    except Exception as e:
        return f"Error: {e}"

def solve_modi(obj_func, constraints):
    try:
        print("MODI Method Started")  # Debugging Step 1

        lines = obj_func.strip().split("\n")
        cost_matrix = [list(map(int, line.split())) for line in lines]

        supply_demand_lines = constraints.strip().split("\n")
        supply = list(map(int, supply_demand_lines[0].split()))
        demand = list(map(int, supply_demand_lines[1].split()))

        print("Cost Matrix:", cost_matrix)  # Debugging Step 2
        print("Supply:", supply)  # Debugging Step 3
        print("Demand:", demand)  # Debugging Step 4

        if sum(supply) != sum(demand):
            return "Error: Supply and Demand must be equal."

        result = modi_algorithm(cost_matrix, supply, demand)

        print("MODI Output:", result)  # Debugging Step 5
        return result

    except Exception as e:
        return f"Error: {e}"


def modi_algorithm(cost_matrix, supply, demand):
    num_sources = len(supply)
    num_destinations = len(demand)

    cost_matrix = np.array(cost_matrix)
    supply = np.array(supply)
    demand = np.array(demand)

    allocation = np.zeros((num_sources, num_destinations), dtype=int)
    remaining_supply = supply.copy()
    remaining_demand = demand.copy()

    print("Starting MODI Algorithm")  # Debugging Step 6

    while np.sum(remaining_supply) > 0 and np.sum(remaining_demand) > 0:
        min_cost = np.min(cost_matrix)
        min_index = np.where(cost_matrix == min_cost)
        row, col = min_index[0][0], min_index[1][0]

        allocation_amount = min(remaining_supply[row], remaining_demand[col])
        allocation[row, col] = allocation_amount

        remaining_supply[row] -= allocation_amount
        remaining_demand[col] -= allocation_amount

        if remaining_supply[row] == 0:
            cost_matrix[row, :] = np.inf
        if remaining_demand[col] == 0:
            cost_matrix[:, col] = np.inf

    total_cost = np.sum(allocation * cost_matrix)

    print("MODI Method Completed")  # Debugging Step 7
    return f"Optimal Cost: {total_cost}, Final Allocation: {allocation.tolist()}"
