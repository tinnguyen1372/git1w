from gprMax.gprMax import api
from gprMax.receivers import Rx
from tools.outputfiles_merge import merge_files
from tools.plot_Bscan import get_output_data, mpl_plot as mpl_plot_Bscan 
from tools.plot_Ascan import mpl_plot as mpl_plot_Ascan
from gprMax.receivers import Rx
import h5py
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
import random
import os
import itertools



class Wall_Func():
    def __init__(self, args) -> None:
        self.args = args
        self.i = args.i
        self.restart = 1
        # self.num_scan = 20
        self.num_scan = 51

        self.resol = 0.005
        self.time_window = 40e-9
        self.square_size = args.square_size
        self.wall_thickness = args.wall_thickness
        # self.wall_height = args.wall_height
        self.wall_permittivity = args.wall_permittivity
        self.wall_conductivity = args.wall_conductivity
        self.object_permittivity = args.object_permittivity
        self.object_conductivity = args.object_conductivity
        # self.object_width = args.obj_width
        # self.object_height = args.obj_height
        self.src_to_wall = 0.10
        self.src_to_rx = 0.05
        # Geometry load
        self.base = os.getcwd() + '/Geometry_ge/Base'
        self.basefile = self.base + '/base{}.png'.format(i)
        self.geofolder = os.getcwd() + '/Geometry_ge/Object'
        self.geofile = self.geofolder + '/geometry{}.png'.format(i)

        # Data load
        self.pix =int(self.square_size/0.005)
        if not os.path.exists('./Input_ge'):
            os.makedirs('./Input_ge')        
        if not os.path.exists('./Input_ge/Base'):
            os.makedirs('./Input_ge/Base')
        if not os.path.exists('./Input_ge/Object'):
            os.makedirs('./Input_ge/Object')
        if not os.path.exists('./Output_ge'):
            os.makedirs('./Output_ge')
        if not os.path.exists('./Output_ge/Base'):
            os.makedirs('./Output_ge/Base')
        if not os.path.exists('./Output_ge/Object'):
            os.makedirs('./Output_ge/Object')
        if not os.path.exists('./Output_ge/WallObj'):
            os.makedirs('./Output_ge/WallObj')
        if not os.path.exists('./BaseImg_ge'):
            os.makedirs('./BaseImg_ge')
        if not os.path.exists('./ObjImg_ge'):
            os.makedirs('./ObjImg_ge')
        if not os.path.exists('./WallObj_ge'):
            os.makedirs('./WallObj_ge')

  
    def view_geometry(self):
        # self.preprocess(self.basefile)
        with h5py.File('./Geometry_ge/geometry_2d.h5', 'r') as f:
            data = f['data'][:]
        
        # Adjust large_array to match data's shape
        data = np.squeeze(data, axis=2)  # Remove any singleton dimensions, if needed
        large_array = np.full(data.shape, -1, dtype=int)
        # Override the values in large_array with data
        large_array[:data.shape[0], :data.shape[1]] = data

        # Mask the regions where the value is 1
        masked_data = ma.masked_where(large_array == -1, large_array)

        # Marker positions based on provided coordinates and scaling factor
        # marker_x, marker_y = 0.15 * data.shape[0] /3.33, 0.15 * data.shape[0] /3.33
        color_list = [
            (1.0, 1.0, 1.0),  # White for -1
            (1.0, 1.0, 0.0),  # Yellow for 0
            (1.0, 0, 0.0)   # Red for 1
        ]
        custom_cmap = ListedColormap(color_list, name="custom_cmap")
        # Plot the markers and masked data
        # plt.plot(marker_x, marker_y, marker='o', color='red', markersize=5)
        plt.imshow(masked_data, cmap='viridis')
        plt.axis('equal')
        plt.title("Geometry Visualization")
        plt.xlabel("X-axis (pixels)")
        plt.ylabel("Y-axis (pixels)")
        plt.show()

    def preprocess(self, filename):
        from PIL import Image
        import numpy as np
        import h5py

        img = Image.open(filename).convert('RGB')  # Convert the image to RGB mode
        # img.show()
        # print(self.pix)
        # Define the color map with a tolerance

        # Base color map
        color_map = {
            (255, 255, 255): -1,  # White (transparent)
            (255, 255, 0): 0,     # Yellow
            (255, 0, 0): 1,       # Red
            (0, 255, 0): 2,       # Green
            (0, 0, 255): 3,       # Blue
        }

        # Limit the dictionary to needed size
        needed_size = len(self.object_permittivity) + 2
        color_map = dict(itertools.islice(color_map.items(), needed_size))

        # Print for debuggings
        # print(color_map)

        def find_most_similar_color(pixel_color, color_map, threshold):
            closest_color = None

            for color, value in color_map.items():
                distance = np.linalg.norm(np.array(pixel_color) - np.array(color))
                if distance < threshold:
                    closest_color = color
            if closest_color is not None:
                return color_map[closest_color]
            else:
                return 0  # Return None when no similar color is found
        # Define the threshold
        threshold = 100  # Adjust this threshold value as needed

        arr_2d = np.zeros((self.pix, self.pix), dtype=int)
        img_resized = img.resize((self.pix, self.pix))
        for y in range(self.pix):
            for x in range(self.pix):
                pixel_color = img_resized.getpixel((x, y))
                arr_2d[y, x] = find_most_similar_color(pixel_color, color_map, threshold)
        arr_2d = np.rot90(arr_2d, k=-1)
        # np.savetxt('output_array.txt', arr_2d, fmt='%d', delimiter=' ')
        self.filename = 'geometry_2d.h5'
        arr_3d = np.expand_dims(arr_2d, axis=2)

        with h5py.File('./Geometry_ge/' + self.filename, 'w') as file:
            dset = file.create_dataset("data", data=arr_3d)
            file.attrs['dx_dy_dz'] = (0.005, 0.005, 0.005)

    def run_base(self):

        # Run gprMax
        self.input = './Input_ge/Base{}.in'.format(self.i)

        pml_cells = 20
        pml = self.resol * pml_cells
        src_to_pml = 0.04

        sharp_domain = self.square_size + 2* self.src_to_rx, self.square_size + 2* self.src_to_rx
        domain_2d = [
            float(sharp_domain[0] + 2 * pml + src_to_pml + 0.2), 
            float(sharp_domain[1] + 2 * pml + src_to_pml + 0.2), 
            0.005
        ]

        # Preprocess geometry

        try:
            with open('{}materials.txt'.format('Base_'), "w") as file:
                file.write('#material: {} {} 1 0 wall\n'.format(self.wall_permittivity, self.wall_conductivity))
            self.preprocess(self.basefile)
        except Exception as e:
            print(e)

        src_position = [pml + src_to_pml + 0.2, 
                        pml + src_to_pml + 0.1, 
                        0]
        rx_position = [pml + src_to_pml + 0.2 + self.src_to_rx, 
                       pml + src_to_pml + 0.1, 
                       0]        
        

        src_steps = [(self.square_size-0.2)/ self.num_scan, 0, 0]
        # print(src_steps)
        config = f'''

#title: Wall Object Imaging

Configuration
#domain: {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f}
#dx_dy_dz: 0.005 0.005 0.005
#time_window: {self.time_window}

#pml_cells: {pml_cells} {pml_cells} 0 {pml_cells} {pml_cells} 0

Source - Receiver - Waveform
#waveform: ricker 1 2e9 my_wave

#hertzian_dipole: z {src_position[0]:.3f} {src_position[1]:.3f} {src_position[2]:.3f} my_wave 
#rx: {rx_position[0]:.3f} {rx_position[1]:.3f} {rx_position[2]:.3f}
#src_steps: {src_steps[0]:.3f} 0 0
#rx_steps: {src_steps[0]:.3f} 0 0

Geometry objects read

#geometry_objects_read: {pml + src_to_pml + 0.1:.3f} {pml+ src_to_pml + 0.2:.3f} {0:.3f} Geometry_ge/geometry_2d.h5 Base_materials.txt
geometry_objects_write: 0 0 0 {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f} Base 
geometry_view: 0 0 0 {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f} 0.005 0.005 0.005 Base n

        '''

        with open(self.input, 'w') as f:
            f.write(config)
            f.close()
        try:
            api(self.input, 
                n=self.num_scan - self.restart + 1, 
                gpu=[0], 
                restart=self.restart,
                geometry_only=False, geometry_fixed=False)
        except Exception as e:
                api(self.input, 
                n=self.num_scan - self.restart + 1, 
                # gpu=[0], 
                restart=self.restart,
                geometry_only=False, geometry_fixed=False)
        
        try:
            merge_files(str(self.input.replace('.in','')), True)
            output_file =str(self.input.replace('.in',''))+ '_merged.out'
            base_output_file = f'./Output_ge/Base/Base{self.i}.out'
            dt = 0

            with h5py.File(output_file, 'r') as f1:
                data1 = f1['rxs']['rx1']['Ez'][()]
                dt = f1.attrs['dt']
                f1.close()

            rxnumber = 1
            rxcomponent = 'Ez'
            plt = mpl_plot_Bscan("merged_output_data", data1, dt, rxnumber,rxcomponent)
            
            fig_width = 15
            fig_height = 15

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            plt.imshow(data1, cmap='gray', aspect='auto')
            plt.axis('off')
            ax.margins(0, 0)  # Remove any extra margins or padding
            fig.tight_layout(pad=0)  # Remove any extra padding


            with h5py.File(base_output_file, 'w') as f_out:
                f_out.attrs['dt'] = dt  # Set the time step attribute
                f_out.create_dataset('rxs/rx1/Ez', data=data1)
                f_out.close()
            plt.savefig(f'./BaseImg_ge/Base{self.i}' + ".png")
            plt.close()
        except Exception as e:
            print(e)


    def run_2D(self):

        # Run gprMax
        self.input = './Input_ge/Object{}.in'.format(self.i)
        pml_cells = 20
        pml = self.resol * pml_cells
        src_to_pml = 0.04

        sharp_domain = self.square_size + 2* self.src_to_rx, self.square_size + 2* self.src_to_rx
        domain_2d = [
            float(sharp_domain[0] + 2 * pml + src_to_pml + 0.2), 
            float(sharp_domain[1] + 2 * pml + src_to_pml + 0.2), 
            0.005
        ]

        # Preprocess geometry

        try:
            with open('{}materials.txt'.format('Obj_'), "w") as file:
                file.write('#material: {} {} 1 0 wall\n'.format(self.wall_permittivity, self.wall_conductivity))
                for i in range(len(self.object_permittivity)):
                    file.write('#material: {} {} 1 0 Object{}\n'.format(self.object_permittivity[i],self.object_conductivity[i],i))          
                self.preprocess(self.geofile)
        except Exception as e:
            print(e)

        src_position = [pml + src_to_pml + 0.2, 
                        pml + src_to_pml + 0.1, 
                        0]
        rx_position = [pml + src_to_pml + 0.2 + self.src_to_rx, 
                       pml + src_to_pml + 0.1, 
                       0]        
        
        src_steps = [(self.square_size-0.2)/ self.num_scan, 0, 0]
        config = f'''

#title: Wall Object Imaging

Configuration
#domain: {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f}
#dx_dy_dz: 0.005 0.005 0.005
#time_window: {self.time_window}

#pml_cells: {pml_cells} {pml_cells} 0 {pml_cells} {pml_cells} 0

Source - Receiver - Waveform
#waveform: ricker 1 2e9 my_wave

#hertzian_dipole: z {src_position[0]:.3f} {src_position[1]:.3f} {src_position[2]:.3f} my_wave 
#rx: {rx_position[0]:.3f} {rx_position[1]:.3f} {rx_position[2]:.3f}
#src_steps: {src_steps[0]:.3f} 0 0
#rx_steps: {src_steps[0]:.3f} 0 0

Geometry objects read

#geometry_objects_read: {pml + src_to_pml + 0.1:.3f} {pml+ src_to_pml + 0.2:.3f} {0:.3f} Geometry_ge/geometry_2d.h5 Obj_materials.txt
geometry_objects_write: 0 0 0 {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f} Object 
geometry_view: 0 0 0 {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f} 0.005 0.005 0.005 Object n

        '''

        with open(self.input, 'w') as f:
            f.write(config)
            f.close()
        try:
            api(self.input, 
                n=self.num_scan - self.restart + 1, 
                gpu=[0], 
                restart=self.restart,
                geometry_only=False, geometry_fixed=False)
        except Exception as e:
                api(self.input, 
                n=self.num_scan - self.restart + 1, 
                # gpu=[0], 
                restart=self.restart,
                geometry_only=False, geometry_fixed=False)
        try:
        
            merge_files(str(self.input.replace('.in','')), True)
            output_file =str(self.input.replace('.in',''))+ '_merged.out'
            uncleaned_output_file = f'./Output_ge/WallObj/Wall_Obj{self.i}.out'
            dt = 0

            with h5py.File(output_file, 'r') as f1:
                data1 = f1['rxs']['rx1']['Ez'][()]
                dt = f1.attrs['dt']
                f1.close()

            # with h5py.File(f'./Output_ge/Base/Base{self.i}.out', 'r') as f1:
            #     data_source = f1['rxs']['rx1']['Ez'][()]

            with h5py.File(uncleaned_output_file, 'w') as f_out:
                f_out.attrs['dt'] = dt  # Set the time step attribute
                f_out.create_dataset('rxs/rx1/Ez', data=data1)
                f_out.close()
            rxnumber = 1
            rxcomponent = 'Ez'
            plt = mpl_plot_Bscan("merged_output_data", data1, dt, rxnumber,rxcomponent)
            fig_width = 15
            fig_height = 15
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            plt.imshow(data1, cmap='gray', aspect='auto')
            plt.axis('off')
            ax.margins(0, 0)  # Remove any extra margins or padding
            fig.tight_layout(pad=0)  # Remove any extra padding
            plt.savefig(f'./WallObj_ge/Wall_Obj{self.i}' + ".png")
            plt.close()
            # data1 = np.subtract(data1, data_source)

            # with h5py.File(output_file, 'w') as f_out:
            #     f_out.attrs['dt'] = dt  # Set the time step attribute
            #     f_out.create_dataset('rxs/rx1/Ez', data=data1)

            # # Draw data with normal plot
            # rxnumber = 1
            # rxcomponent = 'Ez'
            # plt = mpl_plot_Bscan("merged_output_data", data1, dt, rxnumber,rxcomponent)
            
            # fig_width = 15
            # fig_height = 15

            # fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # plt.imshow(data1, cmap='gray', aspect='auto')
            # plt.axis('off')
            # ax.margins(0, 0)  # Remove any extra margins or padding
            # fig.tight_layout(pad=0)  # Remove any extra padding

            # os.rename(output_file, f'./Output_ge/Object/Obj{self.i}.out')
            # plt.savefig(f'./ObjImg_ge/Obj{self.i}' + ".png")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wall Scanning for Through Wall Imaging")      
    parser.add_argument('--start', type=int, default=0, help='Start of the generated geometry')
    parser.add_argument('--end', type=int, default=15, help='End of the generated geometry')
    # data = np.load('SL_Objgeall_0_699.npz', allow_pickle=True)
    # data = np.load('SL_Objgeall_700_1500.npz', allow_pickle=True)
    # data = np.load('Geometry_ge/4w_multi_0_999.npz', allow_pickle=True)
    # data = np.load('Geometry_ge/4w_multi_1000_1999.npz', allow_pickle=True)
    data = np.load('Geometry_ge/4w_multi_0_4999.npz', allow_pickle=True)
    # data = np.load('Geometry_ge/4w_multi_5000_10999.npz', allow_pickle=True)
    datasetvalue = 0
    args = parser.parse_args()
    for i in range(args.start, args.end):
        i = i - datasetvalue
        args.square_size = data['params'][i]['square_size']/100
        args.wall_thickness = data['params'][i]['wall_thickness']/100
        args.wall_permittivity = round(data['params'][i]['permittivity_wall'], 3)
        args.wall_conductivity = round(data['params'][i]['conductivity_wall'], 6)
        args.object_permittivity = [round(p, 3) for p in data['params'][i]['permittivity_object']]
        args.object_conductivity = [round(p, 6) for p in data['params'][i]['conductivity_object']]
        args.i = i + datasetvalue
    # start  adaptor
        wallimg = Wall_Func(args=args)
        print(args)
        # wallimg.view_geometry()
        wallimg.run_base()
        wallimg.run_2D()
