# optimization/views.py
from django.shortcuts import render
from .forms import OptimizationForm
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import linprog

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
            else:
                solution = solve_graphical(obj_func, constraints)
            
            return render(request, 'optimization/result.html', {'solution': solution})
    else:
        form = OptimizationForm()
    return render(request, 'optimization/index.html', {'form': form})

def solve_simplex(obj_func, constraints):
    try:
        # Convert objective function into an array
        c = np.array([float(x) for x in obj_func.strip().split()])  # Strip spaces

        lhs_ineq = []
        rhs_ineq = []

        # Process each constraint
        for constraint in constraints.strip().split("\n"):  # Split constraints line by line
            parts = constraint.strip().split("<=")  # Split by '<='
            if len(parts) != 2:
                return "Error: Invalid constraint format."

            lhs_ineq.append([float(x) for x in parts[0].strip().split()])  # Convert left-hand side to floats
            rhs_ineq.append(float(parts[1].strip()))  # Convert right-hand side to float

        # Convert to numpy arrays
        A_ub = np.array(lhs_ineq)
        b_ub = np.array(rhs_ineq)

        # Solve using Simplex Method (negate c for maximization)
        res = linprog(-c, A_ub=A_ub, b_ub=b_ub, method='simplex')

        return f'Optimal Solution: {res.x}, Objective Value: {-res.fun}'  # Negate result for maximization
    except Exception as e:
        return f'Error: {e}'

def solve_graphical(obj_func, constraints):
    try:
        plt.figure()
        for constraint in constraints.strip().split("\n"):  # Process each constraint
            parts = constraint.strip().split("<=")  # Split by '<='
            if len(parts) != 2:
                return "Error: Invalid constraint format."

            coef = [float(x) for x in parts[0].strip().split()]  # Left-hand side coefficients
            b = float(parts[1].strip())  # Right-hand side value

            x_vals = np.linspace(0, 10, 100)
            y_vals = (b - coef[0] * x_vals) / coef[1]  # Compute y values

            plt.plot(x_vals, y_vals, label=constraint)

        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        plt.title("Graphical Method Solution")
        plt.grid()

        # Ensure static directory exists
        static_dir = os.path.join(os.getcwd(), "static")
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        graph_path = os.path.join(static_dir, "graph.png")
        plt.savefig(graph_path)
        plt.close()

        return '<img src="/static/graph.png" alt="Graphical Solution">'
    except Exception as e:
        return f"Error: {e}"
