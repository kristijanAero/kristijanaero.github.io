# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import json
import random
import logging
import re
from odbAccess import openOdb
from abaqusConstants import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='optimization.log'
)

class Config(object):
    def __init__(self):
        self.BASE_DIR = r"D:\SCIPY"
        self.INP_FILE = os.path.join(self.BASE_DIR, "ISIGHT_STATIC_LINEAR.inp")
        self.MOD_INP_FILE = os.path.join(self.BASE_DIR, "ISIGHT_STATIC_LINEAR_MOD.inp")
        self.ODB_FILE = os.path.join(self.BASE_DIR, "ISIGHT_STATIC_LINEAR.odb")
        self.HISTORY_FILE = os.path.join(self.BASE_DIR, "opti_history.json")
        self.JOB_NAME = "ISIGHT_STATIC_LINEAR"

        self.POP_SIZE = 25
        self.GENERATIONS = 1000
        self.STEP_SIZE = 0.125
        self.ELITE_SIZE = 2
        self.MUTATION_RATE = 0.3

        self.GROUP_BOUNDS = {
            'G1': (0.5, 1.0),
            'G2': (0.375, 0.75),
            'G3': (0.25, 0.5)
        }
        self.C_BOUNDS = {
            0.0: (0.625, 1.875),
            45.0: (0.125, 0.375),
            -45.0: (0.125, 0.375),
            90.0: (0.125, 0.375)
        }

        # Layup groups for S and SS layups, to be updated accordingly
        self.GROUP_S_LAYUPS = {
            'G1': ['S1', 'S2', 'S3', 'S4', 'S1S', 'S2S', 'S3S', 'S4S'],
            'G2': ['S5', 'S6', 'S7', 'S8', 'S5S', 'S6S', 'S7S', 'S8S'],
            'G3': ['S9', 'S10', 'S11', 'S12', 'S9S', 'S10S', 'S11S', 'S12S']
        }

import logging

def modify_inp_file(thickness_sets, config):
    try:
        with open(config.INP_FILE, 'r') as f:
            lines = f.readlines()

        new_lines = []
        i = 0
        detected_layups = []  # Collect detected layup names here

        while i < len(lines):
            line = lines[i]
            new_lines.append(line)

            if line.strip().startswith('*Shell Section') and 'composite' in line.lower():
                section_name_line = lines[i - 1].strip()
                if '** Section:' in section_name_line:
                    section_name = section_name_line.replace('** Section:', '').strip()

                    thickness_key = None

                    # Check in GROUP_S_LAYUPS first
                    for grp, layup_names in config.GROUP_S_LAYUPS.items():
                        if section_name in layup_names:
                            thickness_key = grp
                            break

                    # Detect C1 or C2 layups by name
                    if thickness_key is None:
                        if 'C1' in section_name:
                            thickness_key = 'C1'
                        elif 'C2' in section_name:
                            thickness_key = 'C2'

                    # Detect S and SS layups by naming convention
                    # Infer layup group for S and SS names like S1, S1S, S5-1, S10S-2
                    if thickness_key is None:
                        thickness_key = infer_thickness_group_from_section(section_name)
                    if thickness_key is None:
                        i += 1
                        continue

                    detected_layups.append(section_name)  # Record detected layup

                    layer_set = thickness_sets[thickness_key]

                    # Determine if this is SS layup by naming convention
                    is_SS_layup = section_name.endswith('S') and len(section_name) >= 2 and section_name[-2].isdigit()
                    is_S_layup = (section_name.startswith('S') and not is_SS_layup)

                    i += 1
                    ply_counter = 0
                    ply_thickness_by_group = {}

                    while i < len(lines):
                        ply_line = lines[i].strip()
                        if ply_line.startswith('*') or ply_line.startswith('**'):
                            break

                        parts = ply_line.split(',')
                        if len(parts) >= 4:
                            angle = float(parts[3].strip())
                            ply_counter += 1

                            new_thickness = float(parts[0])

                            if thickness_key in ['C1', 'C2']:
                                if ply_counter <= 8:
                                    if angle == 0.0:
                                        new_thickness = layer_set[1]
                                    elif angle == 90.0:
                                        new_thickness = layer_set[2]
                                    elif angle == 45.0 or angle == -45.0:
                                        new_thickness = layer_set[0]

                            elif thickness_key in ['G1', 'G2', 'G3']:
                                if is_SS_layup and ply_counter <= 4:
                                    # For SS layups, plies 1-4 unchanged
                                    new_thickness = float(parts[0])
                                else:
                                    if is_S_layup:
                                        # Apply symmetries: 1=2=7=8, 3=6, 4=5
                                        if ply_counter in [1, 2, 7, 8]:
                                            sym_key = 'grp1'
                                        elif ply_counter in [3, 6]:
                                            sym_key = 'grp2'
                                        elif ply_counter in [4, 5]:
                                            sym_key = 'grp3'
                                        else:
                                            sym_key = 'unique_%d' % ply_counter

                                        if sym_key not in ply_thickness_by_group:
                                            if angle == 0.0:
                                                ply_thickness_by_group[sym_key] = layer_set[1]
                                            elif angle == 90.0:
                                                ply_thickness_by_group[sym_key] = layer_set[2]
                                            elif angle == 45.0 or angle == -45.0:
                                                ply_thickness_by_group[sym_key] = layer_set[0]
                                        new_thickness = ply_thickness_by_group[sym_key]
                                    else:
                                        # Normal G layup: all plies changed
                                        if angle == 0.0:
                                            new_thickness = layer_set[1]
                                        elif angle == 90.0:
                                            new_thickness = layer_set[2]
                                        elif angle == 45.0 or angle == -45.0:
                                            new_thickness = layer_set[0]

                            parts[0] = "{:.3f}".format(new_thickness)
                            new_lines.append(', '.join(parts) + '\n')
                        else:
                            new_lines.append(lines[i])

                        i += 1
                    continue
            i += 1

        # Print all detected layups before writing the modified file
        print("Detected layups to be modified:")
        for layup in detected_layups:
            print (" -", layup)

        with open(config.MOD_INP_FILE, 'w') as f:
            f.writelines(new_lines)

    except IOError as e:
        logging.error("Error modifying input file: %s", str(e))
        raise

def infer_thickness_group_from_section(section_name):
    """
    Infers G1, G2, or G3 from S or SS layup section name like S1, S1S, S9-1, S12S-1, etc.
    Returns group name or None if not matched.
    """
    match_s = re.match(r'^S([1-9]|1[0-2])(-\d+)?$', section_name)
    match_ss = re.match(r'^S([1-9]|1[0-2])S(-\d+)?$', section_name)
    if match_s or match_ss:
        num = int(match_s.group(1)) if match_s else int(match_ss.group(1))
        if 1 <= num <= 4:
            return 'G1'
        elif 5 <= num <= 8:
            return 'G2'
        elif 9 <= num <= 12:
            return 'G3'
    return None    

def run_abaqus_job(config):
    cmd = 'abaqus job={} input={} interactive'.format(
        config.JOB_NAME,
        os.path.basename(config.MOD_INP_FILE)
    )
    logging.info("Running Abaqus job: %s", cmd)
    os.system(cmd)

def extract_results(config):
    try:
        odb = openOdb(config.ODB_FILE, readOnly=True)
        step_keys = odb.steps.keys()
        if not step_keys:
            raise ValueError("No steps found in ODB file.")
        last_step_key = step_keys[-1]
        last_step = odb.steps[last_step_key]
        if not last_step.frames:
            raise ValueError("No frames found in the last step.")
        last_frame = last_step.frames[-1]

        if 'TSAIW' not in last_frame.fieldOutputs:
            raise KeyError("TSAIW field output not found in the last frame.")
        tsai_field = last_frame.fieldOutputs['TSAIW']
        max_tsai = max([value.data for value in tsai_field.values])

        history_regions = odb.steps[last_step_key].historyRegions
        target_history_region = None
        for region_key in history_regions.keys():
            if 'ASSEMBLY' in region_key.upper():
                target_history_region = history_regions[region_key]
                break

        if not target_history_region:
            raise KeyError("No suitable history region found for mass extraction.")

        mass_properties = target_history_region.historyOutputs
        target_mass_output_key = None
        for output_key in mass_properties.keys():
            if 'MASS' in output_key.upper():
                target_mass_output_key = output_key
                break

        if not target_mass_output_key:
            raise KeyError("No suitable mass output found.")

        total_mass = mass_properties[target_mass_output_key].data[-1][1]

        odb.close()

        logging.info("Extracted TSAIW: %.4f, Mass: %.4f" % (max_tsai, total_mass))
        return max_tsai, total_mass

    except Exception as e:
        logging.error("Failed to extract results from ODB: %s", str(e))
        raise

class GeneticOptimizer(object):
    def __init__(self, config):
        self.config = config
        self.best_solution = None
        self.history = []
        self.processed_individuals = set()
        self.run_counter = 0

        if os.path.exists(config.HISTORY_FILE):
            try:
                with open(config.HISTORY_FILE, 'r') as f:
                    self.history = json.load(f)
                    if self.history:
                        self.best_solution = min(self.history, key=lambda x: x['fitness'])
                        for entry in self.history:
                            self.processed_individuals.add(self._flatten_individual(entry['thickness_sets']))
                logging.info("Loaded previous optimization history")
            except Exception as e:
                logging.warning("Could not load previous history: %s", str(e))

    def generate_individual(self):
        def generate_group(group_key):
            bounds = self.config.GROUP_BOUNDS[group_key]
            return [round(random.uniform(*bounds) / self.config.STEP_SIZE) * self.config.STEP_SIZE for _ in range(3)]

        def generate_C(group_key):
            bounds_45 = self.config.C_BOUNDS[45.0]  # bounds for ±45°
            bounds_0 = self.config.C_BOUNDS[0.0]    # bounds for 0°
            bounds_90 = self.config.C_BOUNDS[90.0]  # bounds for 90°
        
            return [
                round(random.uniform(*bounds_45) / self.config.STEP_SIZE) * self.config.STEP_SIZE,  # ±45°
                round(random.uniform(*bounds_0) / self.config.STEP_SIZE) * self.config.STEP_SIZE,   # 0°
                round(random.uniform(*bounds_90) / self.config.STEP_SIZE) * self.config.STEP_SIZE   # 90°
            ]

        individual = {
            'G1': generate_group('G1'),
            'G2': generate_group('G2'),
            'G3': generate_group('G3'),
            'C1': generate_C('C1'),
            'C2': generate_C('C2')
        }
        return individual

    def _flatten_individual(self, thickness_sets):
        flat = []
        for key in ['G1', 'G2', 'G3', 'C1', 'C2']:
            flat.extend([round(t, 3) for t in thickness_sets[key]])
        return tuple(flat)

    def fitness(self, thickness_sets):
        self.run_counter += 1
        print("Run #{}".format(self.run_counter))


        total_thickness = {}
        for grp in ['G1', 'G2', 'G3', 'C1', 'C2']:
            avg_thickness = sum(thickness_sets[grp])
            total_thickness[grp] = avg_thickness

        if not (total_thickness['C1'] > total_thickness['C2'] > total_thickness['G1'] > total_thickness['G2'] > total_thickness['G3']):
            print("Thickness constraints violated")
            return 1e9

        for cgrp in ['C1', 'C2']:
            ply_45_and_m45, ply_0, ply_90 = thickness_sets[cgrp]
            if not (ply_0 > ply_45_and_m45 and ply_0 > ply_90):
                print("0-degree ply thickness rule violated in {}: {}".format(cgrp, thickness_sets[cgrp]))
                return 1e9

        flat = self._flatten_individual(thickness_sets)
        if flat in self.processed_individuals:
            return None
        self.processed_individuals.add(flat)

        modify_inp_file(thickness_sets, self.config)
        run_abaqus_job(self.config)
        tsaiw, mass = extract_results(self.config)
        print("Extracted TSAIW: {}, Mass: {}".format(tsaiw, mass))


        fitness_val = tsaiw + 0.1 * mass

        print("Thickness sets and total thickness sums:")
        for grp in ['G1', 'G2', 'G3', 'C1', 'C2']:
            print("{}: {}, Total Thickness: {:.4f}".format(grp, thickness_sets[grp], total_thickness[grp]))
            print("TSAIW: {:.4f}, Mass: {:.4f}".format(tsaiw, mass))


        self.history.append({
            'thickness_sets': thickness_sets.copy(),
            'tsaiw': tsaiw,
            'mass': mass,
            'fitness': fitness_val
        })

        with open(self.config.HISTORY_FILE, 'w') as f:
            json.dump(self.history, f, indent=4)

        if self.best_solution is None or fitness_val < self.best_solution['fitness']:
            self.best_solution = {'thickness_sets': thickness_sets.copy(), 'fitness': fitness_val}
            logging.info("New best fitness: {:.4f}".format(fitness_val))


        return fitness_val

    # Crossover between two parents to produce a child
    def crossover(self, parent1, parent2):
        child = {}
        for key in parent1.keys():
            # Randomly choose gene (thickness set) from either parent
            if random.random() < 0.5:
                child[key] = parent1[key][:]
            else:
                child[key] = parent2[key][:]
        return child

    # Mutation operator: randomly mutate values in thickness sets
    def mutate(self, thickness_sets, mutation_rate):
        new_sets = {}
        for key, values in thickness_sets.items():
            new_vals = []
            if key in ['G1', 'G2', 'G3']:
                bounds = self.config.GROUP_BOUNDS[key]
                for val in values:
                    if random.random() < mutation_rate:
                        step = self.config.STEP_SIZE
                        delta = random.choice([-step, step])
                        new_val = round(val + delta, 3)
                        # Clamp to group-specific bounds
                        new_val = max(bounds[0], min(bounds[1], new_val))
                        new_vals.append(new_val)
                    else:
                        new_vals.append(val)

            elif key in ['C1', 'C2']:
                # Ply angles order: ±45°, 0°, 90°
                ply_angles = [45.0, 0.0, 90.0]

                for i, val in enumerate(values):
                    bounds = self.config.C_BOUNDS[ply_angles[i]]
                    if random.random() < mutation_rate:
                        step = self.config.STEP_SIZE
                        delta = random.choice([-step, step])
                        new_val = round(val + delta, 3)
                        # Clamp to angle-specific bounds
                        new_val = max(bounds[0], min(bounds[1], new_val))
                        new_vals.append(new_val)
                    else:
                        new_vals.append(val)

            new_sets[key] = new_vals
        return new_sets

    def optimize(self):
        # Initialize population
        population = []
        while len(population) < self.config.POP_SIZE:
            indiv = self.generate_individual()
            fitness_val = self.fitness(indiv)
            if fitness_val is not None:
                population.append({'indiv': indiv, 'fitness': fitness_val})

        for gen in range(self.config.GENERATIONS):
            print("Generation {}/{}".format(gen+1, self.config.GENERATIONS))




            # Sort population by fitness ascending (lower is better)
            population.sort(key=lambda x: x['fitness'])

            # Save elites
            elites = population[:self.config.ELITE_SIZE]

            # Adaptive mutation rate: decrease over generations
            mutation_rate = max(0.05, self.config.MUTATION_RATE * (1 - gen / self.config.GENERATIONS))
            print("Mutation rate: {:.3f}".format(mutation_rate))


            # Create new population starting with elites
            new_population = elites[:]

            # Generate offspring until population is replenished
            while len(new_population) < self.config.POP_SIZE:
                # Select parents via tournament selection (size 3)
                parents = []
                for _ in range(2):
                    tournament = random.sample(population, 3)
                    tournament.sort(key=lambda x: x['fitness'])
                    parents.append(tournament[0]['indiv'])

                # Crossover
                child = self.crossover(parents[0], parents[1])

                # Mutation
                child = self.mutate(child, mutation_rate)

                # Evaluate fitness
                fitness_val = self.fitness(child)
                if fitness_val is not None:
                    new_population.append({'indiv': child, 'fitness': fitness_val})

            population = new_population

            # Update best solution found
            current_best = min(population, key=lambda x: x['fitness'])
            if self.best_solution is None or current_best['fitness'] < self.best_solution['fitness']:
                self.best_solution = {'thickness_sets': current_best['indiv'], 'fitness': current_best['fitness']}
                print("New best fitness at generation {}: {:.4f}".format(gen + 1, self.best_solution['fitness']))


        print("Optimization finished.")
        print("Best fitness: {:.4f}".format(self.best_solution['fitness']))
        print("Best thickness sets:")
        for k, v in self.best_solution['thickness_sets'].items():
            print("{}: {}".format(k, v))


def main():
    config = Config()
    optimizer = GeneticOptimizer(config)
    optimizer.optimize()

if __name__ == '__main__':
    main()
