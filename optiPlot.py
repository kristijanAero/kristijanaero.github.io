# -*- coding: utf-8 -*-
from __future__ import print_function
import json
import matplotlib.pyplot as plt

def plot_tsaiwu_vs_mass(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    tsaiwu_values = []
    mass_values = []

    for entry in data:
        tsaiwu_values.append(entry.get('tsaiwu'))
        mass_values.append(entry.get('mass'))

    plt.figure(figsize=(8,6))
    plt.scatter(mass_values, tsaiwu_values, c='blue', marker='o')
    plt.xlabel('Mass')
    plt.ylabel('TSAIW')
    plt.title('TSAIW vs Mass from Optimization History')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    json_file = 'D:\SCIPY\opti_history.json'  # Change this path if needed
    plot_tsaiwu_vs_mass(json_file)
