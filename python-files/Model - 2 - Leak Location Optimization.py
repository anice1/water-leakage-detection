#!/usr/bin/env python
# coding: utf-8

# # Leak Location Optimization
# ### Import Necessary Libraries



import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import pygad
import numpy
import glob
import wntr
import os

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12,8)


class Config:
    num_generations =100 #Number of generations.
    num_parents_mating = 1 # Number of solutions to be selected as parents in the mating pool.
    sol_per_pop = 20 # Number of solutions in the population.
    num_genes = 1 #len(function_inputs) --> this can be an array
    init_range_low = 0
    init_range_high = 1
    parent_selection_type = "sss" # Type of parent selection.
    keep_parents = -1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.
    crossover_type = "single_point" # Type of the crossover operator. 
    mutation_type = "random" # Type of the mutation operator.
    mutation_num_genes=1 # Type of the mutation operator.
    last_fitness = 0
    
    INP_FILE_PATH = r"../data/Real_Synthetic_Net.inp" 
    TEMP_DIR = "../temp/"
    PLOTS_DIR = "../plots"
    FUNCTION_INPUTS = None
    DESIRED_OUTPUT = None


class WaterLeakModel(Config):
    
    def __init__(self, node:str):
        ''' 
            node: node/junction ID in the network
        '''
        
        # Check if node is junction_name 
        wn = wntr.network.WaterNetworkModel(self.INP_FILE_PATH)
        assert wn.junction_name_list.count(node) == 1, "Node not found in the network"
        self.node_index = wn.junction_name_list.index(node)
        
        self.node = node
        
    
    def simulate(self, wn, plot_graph=True):
        """
        If plot_graph is set to true, a graph of the network is plotted after simulation
        """
        # Plot pressure after add leak to node
            
        sim = wntr.sim.WNTRSimulator(wn)
        results = sim.run_sim()
        pressure = results.node['pressure']
        pressure_at_N6 = pressure.loc[:,wn.junction_name_list[self.node_index]]
        if plot_graph:
            wntr.graphics.plot_network(wn, node_attribute=pressure.any(), node_size=150, title='Pressure at 0 hours')
        
        return results
    
    def change_discharge_coefficient(self, wn, emitter_value):
        """
        This function changes the emitter coefficient for selected node in the network, and create a 
        structured representation of our data, a csv.
        
        parameters
        ----------
        wn: wntr network object
        emitter_value: the emitter coefficient value
        """
        
        # Change emitter coefficient
        node = wn.get_node(str(self.node))
        node.add_leak(wn, area=0.00015, start_time=0, end_time=1, discharge_coeff=emitter_value)
        return self.simulate(wn, plot_graph=False)
    
    def export_results(self, solutions:list, path:str, name:str):
        """
        Concatenates all solutions generated and exports as a single csv file
        
        parameters
        ----------
        solution: a list of all paths to the solutions csv files
        path: path where concatenated solution will be exported to
        name: name to be assigned to exported file
        """
        if not os.path.exists(path):
            os.mkdir(path)
        
        temp = pd.DataFrame()
        for i in range(len(solutions)):
            data = pd.read_csv(solutions[i])
            if i == 0:
                temp = data
            else:
                temp = pd.concat([temp, data])
        name+=".csv"
        try:
            temp.to_csv(os.path.join(path,name),index=False)
            print(f'File Exported Successfully to path: {os.path.join(path, name)}')
        except Exception as e:
            print(e)
            
    def run(self, leak_area=0.00015, start_time=0, end_time=1, discharge_coeff=.5, function_inputs=0.5, plot_graph=True):
        """
        Adds a leak to node passed to WaterLeakModel() object and simulates
        
        parameters
        ----------
        leak_area: area of the leak
        start_time: time in seconds to start the leak
        end_time: time in seconds to end the leak
        discharge_coeff: Leak discharge coefficient; Takes on values between 0 and 1.
        function_inputs = inputs for optimization, can be array of numbers
        plot_graph: If plot_graph is set to true, a graph of the network is plotted after simulation
        """
        # Add leak and simulate
        wn = wntr.network.WaterNetworkModel(self.INP_FILE_PATH)
        node = wn.get_node(self.node)
        node.add_leak(wn, area=leak_area, start_time=start_time, end_time=end_time, discharge_coeff=.5)
        self.simulate(wn, plot_graph=plot_graph)

        self.FUNCTION_INPUTS = function_inputs
        self.DESIRED_OUTPUT = node.head - node.elevation



# Instantiate the pygad optimization class
water_model = WaterLeakModel(node='N6')
water_model.run()


# > **The node with the highest fitness is considered to have a leak.**
desired_output = 26.5
function_input = 0.5

wn = wntr.network.WaterNetworkModel(water_model.INP_FILE_PATH)
all_nodes = wn.junction_name_list
data=pd.DataFrame(columns=['NODE','EMITTER_COEFFICIENT','PRESSURE_OUTPUT','FITNESS']) 


for i in range(len(all_nodes)):
    wn = wntr.network.WaterNetworkModel(water_model.INP_FILE_PATH)
    water_model = WaterLeakModel(node=wn.junction_name_list[i])
    results = water_model.change_discharge_coefficient(wn, emitter_value=function_input)
    pressure = results.node['pressure']
    pressure_output = pressure.loc[:,water_model.node]
    fitness = 1.0 / (np.abs(pressure_output - desired_output) + 0.000001)
    
    data=data.append({'NODE':all_nodes[i],'EMITTER_COEFFICIENT':function_input,'PRESSURE_OUTPUT':list(pressure_output)[0],'FITNESS':list(fitness)[0]},ignore_index=True)
    data=data.sort_values(by='FITNESS',ascending=False)
    data=data.reset_index(drop=True)


plt.figure(figsize=(10,5))
plt.bar(data['NODE'],data['FITNESS'])

plt.xlabel('NODE ',fontsize=14)
plt.ylabel('FITNESS',fontsize=14)

plt.title('Fitness Across Nodes',fontsize=16)
plt.savefig(f"{water_model.PLOTS_DIR}/node_fitness.png")


# After our optimization it was proven that the leak node was `NODE 6` since node 6 has the highest fitness

plt.figure(figsize=(10,5))
plt.plot(data['NODE'],data['PRESSURE_OUTPUT'])

plt.xlabel('NODE ',fontsize=14)
plt.ylabel('PRESSURE',fontsize=14)

plt.title('Pressure Across Nodes',fontsize=16)
plt.savefig(f"{water_model.PLOTS_DIR}/node_pressure.png")
