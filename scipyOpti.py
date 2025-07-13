# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import json
import random
import logging
from odbAccess import openOdb
from abaqusConstants import *

# Configure logging
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
        self.MUTATION_RATE = 0.15

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

def modify_inp_file(thickness_sets, config):
    try:
        with open(config.INP_FILE, 'r') as f:
            lines = f.readlines()

        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            new_lines.append(line)

            if line.strip().startswith('*Shell Section') and 'composite' in line.lower():
                section_name_line = lines[i - 1].strip()
                if '** Section:' in section_name_line:
                    section_name = section_name_line.replace('** Section:', '').strip()

                    if 'C1' in section_name:
                        layer_set = thickness_sets['C1']
                    elif 'C2' in section_name:
                        layer_set = thickness_sets['C2']
                    elif 'G1' in section_name:
                        layer_set = thickness_sets['G1']
                    elif 'G2' in section_name:
                        layer_set = thickness_sets['G2']
                    elif 'G3' in section_name:
                        layer_set = thickness_sets['G3']
                    else:
                        i += 1
                        continue

                    i += 1
                    while i < len(lines):
                        ply_line = lines[i].strip()
                        if ply_line.startswith('*') or ply_line.startswith('**'):
                            break

                        parts = ply_line.split(',')
                        if len(parts) >= 4:
                            angle = float(parts[3].strip())

                            if angle == 0.0:
                                thickness = layer_set[2]
                            elif angle == 45.0:
                                thickness = layer_set[0]
                            elif angle == -45.0:
                                thickness = layer_set[1]
                            elif angle == 90.0:
                                thickness = layer_set[2]
                            else:
                                thickness = parts[0]

                            parts[0] = "{:.3f}".format(thickness)
                            new_lines.append(', '.join(parts) + '\n')
                        else:
                            new_lines.append(lines[i])

                        i += 1
                    continue

            i += 1

        with open(config.MOD_INP_FILE, 'w') as f:
            f.writelines(new_lines)

    except IOError as e:
        logging.error("Error modifying input file: %s", str(e))
        raise

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
        return {
            'C1': [round(random.uniform(*self.config.C_BOUNDS[0.0]) / self.config.STEP_SIZE) * self.config.STEP_SIZE for _ in range(3)],
            'C2': [round(random.uniform(*self.config.C_BOUNDS[45.0]) / self.config.STEP_SIZE) * self.config.STEP_SIZE for _ in range(3)],
            'G1': [round(random.uniform(*self.config.GROUP_BOUNDS['G1']) / self.config.STEP_SIZE) * self.config.STEP_SIZE for _ in range(3)],
            'G2': [round(random.uniform(*self.config.GROUP_BOUNDS['G2']) / self.config.STEP_SIZE) * self.config.STEP_SIZE for _ in range(3)],
            'G3': [round(random.uniform(*self.config.GROUP_BOUNDS['G3']) / self.config.STEP_SIZE) * self.config.STEP_SIZE for _ in range(3)]
        }

    def crossover(self, parent1, parent2):
        child = {}
        for k in parent1:
            child[k] = [p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1[k], parent2[k])]
        return child

    def mutate(self, individual):
        for k in individual:
            bounds = (self.config.C_BOUNDS[0.0] if k == 'C1' else
                      self.config.C_BOUNDS[45.0] if k == 'C2' else
                      self.config.GROUP_BOUNDS[k])
            for i in range(len(individual[k])):
                if random.random() < self.config.MUTATION_RATE:
                    individual[k][i] = round(random.uniform(*bounds) / self.config.STEP_SIZE) * self.config.STEP_SIZE
        return individual

    def _flatten_individual(self, ind):
        # Flatten individual to a tuple for uniqueness check
        # Sort keys to maintain consistent order
        return tuple(sorted((k, tuple(v)) for k, v in ind.items()))

    def is_unique(self, ind):
        flattened_ind = self._flatten_individual(ind)
        return flattened_ind not in self.processed_individuals

    def save_history(self):
        try:
            with open(self.config.HISTORY_FILE, 'w') as f:
                json.dump(self.history, f, indent=4)
        except Exception as e:
            logging.error("Failed to save history: %s", str(e))

    def optimize(self):
        population = [self.generate_individual() for _ in range(self.config.POP_SIZE)]

        for gen in range(self.config.GENERATIONS):
            print("\nGeneration {}/{}".format(gen + 1, self.config.GENERATIONS))
            results = []

            for idx, ind in enumerate(population):
                print("\nRun {}/{}".format(idx + 1, self.config.POP_SIZE))
                print("Thickness sets:")
                for k, v in ind.items():
                    print("{} = {}".format(k, v))

                if not self.is_unique(ind):
                    logging.info("Skipping duplicate individual: %s", ind)
                    print("Skipping duplicate individual.")
                    results.append({'individual': ind, 'tsaiwu': 1e9, 'mass': 1e9, 'fitness': 2e9})
                    continue

                try:
                    modify_inp_file(ind, self.config)
                    run_abaqus_job(self.config)
                    tsaiwu, mass = extract_results(self.config)

                    print("TSAIW: {:.4f}".format(tsaiwu))
                    print("Mass: {:.4f}".format(mass))

                    fitness = mass if tsaiwu < 1.0 else 1e6 + mass
                    results.append({'individual': ind, 'tsaiwu': tsaiwu, 'mass': mass, 'fitness': fitness})

                    self.processed_individuals.add(self._flatten_individual(ind))

                    self.history.append({
                        'generation': gen + 1,
                        'run': idx + 1,
                        'thickness_sets': ind,
                        'tsaiwu': float(tsaiwu),
                        'mass': float(mass),
                        'fitness': float(fitness)
                    })

                    if (not self.best_solution) or (fitness < self.best_solution['fitness']):
                        self.best_solution = results[-1]

                    self.save_history()

                except Exception as e:
                    logging.error("Error in optimization: %s", str(e))
                    print("Error during optimization run: {}".format(str(e)))
                    results.append({'individual': ind, 'tsaiwu': 1e9, 'mass': 1e9, 'fitness': 2e9})
                    continue

            results.sort(key=lambda x: x['fitness'])
            elite = [r['individual'] for r in results[:self.config.ELITE_SIZE]]
            parents = [r['individual'] for r in results[:self.config.POP_SIZE // 2]]

            population = elite[:]
            while len(population) < self.config.POP_SIZE:
                p1, p2 = random.sample(parents, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                population.append(child)

def main():
    config = Config()
    optimizer = GeneticOptimizer(config)
    optimizer.optimize()

    if optimizer.best_solution:
        print("\nBest solution found:")
        for k, v in optimizer.best_solution['individual'].items():
            print("{} = {}".format(k, v))
        print("TSAIW: {:.4f}".format(optimizer.best_solution['tsaiwu']))
        print("Mass: {:.4f}".format(optimizer.best_solution['mass']))

if __name__ == '__main__':
    main()
