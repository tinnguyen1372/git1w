import os
import numpy as np
import h5py
from tools.plot_Bscan import get_output_data, mpl_plot as mpl_plot_Bscan 
import matplotlib.pyplot as plt

base_folder = './Output_ge/Base'
output_folder = './Output_ge/WallObj'
output_img_folder = './ObjImg_ge'
output_data_folder = './Output_ge/Object'

if not os.path.exists(output_img_folder):
    os.makedirs(output_img_folder)

if not os.path.exists(output_data_folder):
    os.makedirs(output_data_folder)

for filename in os.listdir(output_folder):
    if filename.endswith('.out'):
        # print(f'Base{filename.split("Wall_Obj")[1]}')
        base_file = os.path.join(base_folder,f'Base{filename.split("Wall_Obj")[1]}')
        output_file = os.path.join(output_folder, filename)
        
        with h5py.File(base_file, 'r') as f_base, h5py.File(output_file, 'r') as f_out:
            data_base = f_base['rxs/rx1/Ez'][:]
            data_out = f_out['rxs/rx1/Ez'][:]
            dt = f_out.attrs['dt']
        
        data1 = np.subtract(data_out, data_base)

        rxnumber = 1
        rxcomponent = 'Ez'
        plt = mpl_plot_Bscan("merged_output_data", data1, dt, rxnumber, rxcomponent)
        
        fig_width = 15
        fig_height = 15
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        plt.imshow(data1, cmap='gray', aspect='auto')
        plt.axis('off')
        ax.margins(0, 0)
        fig.tight_layout(pad=0)
        
        img_filename = os.path.join(output_img_folder, f'Obj{filename.split("Wall_Obj")[1]}.png')
        plt.savefig(img_filename)
        
        new_output_file = os.path.join(output_data_folder, f'Obj{filename.split("Wall_Obj")[1]}')
        with h5py.File(new_output_file, 'w') as f_new_out:
            f_new_out.attrs['dt'] = dt
            f_new_out.create_dataset('rxs/rx1/Ez', data=data1)