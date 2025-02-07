# optimization/views.py
from django.shortcuts import render
from .forms import OptimizationForm
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import linprog
from django.shortcuts import render

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
        return "MODI method implementation coming soon."
    except Exception as e:
        return f"Error: {e}"
