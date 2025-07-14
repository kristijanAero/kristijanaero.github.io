# -*- coding: utf-8 -*-
import json
import os
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for compatibility with various environments
import matplotlib.pyplot as plt

def plot_tsaiw_vs_mass(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    tsaiw_values = []
    mass_values = []

    for entry in data:
        tsaiw_values.append(entry.get('tsaiw'))
        mass_values.append(entry.get('mass'))

    plt.figure(figsize=(8, 6))
    plt.scatter(mass_values, tsaiw_values, c='blue', marker='o')
    plt.xlabel('Mass')
    plt.ylabel('TSAIW')
    plt.title('TSAIW vs Mass from Optimization History')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    json_file = r'D:\SCIPY\opti_history.json'  # Adjust path if needed
    if os.path.exists(json_file):
        plot_tsaiw_vs_mass(json_file)
    else:
        print(f"JSON file not found: {json_file}")
