import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
from matplotlib.colors import ListedColormap
from scipy.ndimage import rotate
import os
from scipy.ndimage import convolve

def create_geometry(square_size, air_size, wall_thickness):
    # Initialize the square room with walls
    geometry = np.ones((square_size, square_size), dtype=int)

    # Set the air region
    air_start = (square_size - air_size) // 2
    air_end = air_start + air_size
    geometry[air_start:air_end - wall_thickness, air_start:air_end] = 0  # Air is represented by 1

    return geometry, air_start, air_end
def get_random_material():
    materials = {
        "Concrete": {"type": "Dielectric", "permittivity": 5.24, "conductivity": 0.001},
        "Brick": {"type": "Dielectric", "permittivity": 3.91, "conductivity": 0.002},
        "Plasterboard": {"type": "Dielectric", "permittivity": 2.73, "conductivity": 0.0005},
        "Wood": {"type": "Dielectric", "permittivity": 1.99, "conductivity": 0.0002},
        "Glass": {"type": "Dielectric", "permittivity": 6.31, "conductivity": 0.00001},
        "Aluminum": {"type": "Metallic", "permittivity": 1, "conductivity": 3.77e7},
        "Copper": {"type": "Metallic", "permittivity": 1, "conductivity": 5.8e7},
        "Gold": {"type": "Metallic", "permittivity": 1, "conductivity": 4.1e7},
        "Silver": {"type": "Metallic", "permittivity": 1, "conductivity": 6.3e7},
        "Iron": {"type": "Metallic", "permittivity": 1, "conductivity": 1e7},
        "Dry Soil": {"type": "Nonmetallic", "permittivity": 4.0, "conductivity": 0.001},
        "Ice": {"type": "Nonmetallic", "permittivity": 3.2, "conductivity": 0.00001},
    }
    
    variance_factor = 0.15
    material = random.choice(list(materials.keys()))

    if materials[material]["type"] == "Metallic":
        permittivity = materials[material]["permittivity"]
    else:
        permittivity = materials[material]["permittivity"] * random.uniform(1 - variance_factor, 1 + variance_factor)
    conductivity = materials[material]["conductivity"] * random.uniform(1 - variance_factor, 1 + variance_factor)
    return material, materials[material]["type"], round(permittivity, 3), round(conductivity, 6)

def add_random_shape(i, geometry, air_start, air_end, wall_thickness):
    obj_mat,obj_type,permittivity_object, conductivity_object = get_random_material()
    objwall_gap = 25  # Gap between object and wall
    shape = random.choice(["rectangle", "triangle", "circle"])
    rect_width = random.randint(20, 40)
    rect_height = random.randint(20, 40)
    # rect_y = random.randint(air_start + wall_thickness + objwall_gap, air_end - rect_height - square_size//4)
    # # rect_x = random.randint(air_start + int(6*square_size/22), air_end - rect_width - int(6*square_size/22))
    # rect_x = 200

    if i == 0:
        rect_y = random.randint(105, air_end - rect_height - square_size//4)
        rect_x = random.randint(air_start + int(6*square_size/22), air_end - rect_width - int(6*square_size/22))
    elif i == 1:
        rect_y = random.randint(air_start + wall_thickness + objwall_gap, 100 - rect_height)
        rect_x = random.randint(air_start + int(6*square_size/22), 100 - rect_width)
    elif i == 2:
        rect_y = random.randint(air_start + wall_thickness + objwall_gap, 100 - rect_height)
        rect_x = random.randint(105, air_end - rect_width - int(5*square_size/22))

    # rotation_angle = random.randint(0, 360)
    rotation_angle = 0

    # Define a blank canvas for the shape
    shape_canvas = np.zeros_like(geometry)

    if shape == "rectangle":
        shape_canvas[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width] = int(i) + 2

    elif shape == "triangle":
        for y in range(rect_height):
            for x in range(rect_width - y):
                shape_canvas[rect_y + y, rect_x + x] = int(i) + 2

    elif shape == "circle":
        radius = min(rect_width, rect_height) // 2
        x_center, y_center = rect_x + radius, rect_y + radius
        for y in range(-radius, radius):
            for x in range(-radius, radius):
                if x**2 + y**2 <= radius**2:
                    shape_canvas[y_center + y, x_center + x] = int(i) + 2

    # Rotate the shape
    rotated_shape = rotate(shape_canvas, angle=rotation_angle, reshape=False, order=0)


    # Define a 5x5 kernel for checking neighborhood
    kernel = np.ones((10, 10), dtype=int)

    # Create a mask where nonzero values exist in a 5x5 neighborhood
    nonzero_mask = convolve((geometry > 0).astype(int), kernel, mode='constant', cval=0) > 0

    # Valid positions are where geometry is 0 and no nonzero values exist in a 5x5 region
    valid_positions = (geometry == 0) & (~nonzero_mask)

    # Add the shape only in valid positions
    geometry[valid_positions & (rotated_shape > 1)] = rotated_shape[valid_positions & (rotated_shape > 1)]

    return obj_mat,obj_type,permittivity_object, conductivity_object, shape, geometry

def visualize_geometry(geometry, wall_color, air_color, f_color, s_color, t_color):
    cmap = ListedColormap([
        air_color, 
        wall_color, 
        f_color, 
        s_color, 
        t_color
    ])
    plt.figure(figsize=(10, 10))
    plt.imshow(geometry, cmap=cmap, origin='lower')
    plt.title('Geometry with Square Walls and Random Shapes')
    plt.axis('off')
    plt.show()

def save_image(filename, numobjects, geometry, square_size, wall_color, air_color, f_color, s_color, t_color):
    cmap = ListedColormap([
        air_color, 
        wall_color, 
        f_color,
        s_color,
        t_color
    ][:numobjects + 2])
    plt.figure(figsize=(10, 10))
    plt.imshow(geometry, cmap=cmap)
    plt.xlabel('x(t)')
    plt.ylabel('y(t)')
    plt.axis('off')    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(filename, format='png', dpi=square_size / 10, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_base(filename, geometry, square_size, wall_color, air_color):
    cmap = ListedColormap([air_color,wall_color])    
    plt.figure(figsize=(10, 10))
    plt.imshow(geometry, cmap=cmap)
    plt.xlabel('x(t)')
    plt.ylabel('y(t)')
    plt.axis('off')    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(filename, format='png', dpi=square_size / 10, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_parameters(filename, **params):
    if os.path.exists(filename):
        existing_data = np.load(filename, allow_pickle=True)
        all_params = list(existing_data['params'])
    else:
        all_params = []

    all_params.append(params)

    with open(filename, 'wb') as f:
        np.savez(f, params=all_params)

if __name__ == '__main__':
    # Predefined colors
    wall_color = [1, 1, 0]   # Wall color
    air_color = [1, 1, 1]    # Air color
    f_color = [1, 0, 0]      # First shape color
    s_color = [0, 1, 0]      # Second shape color
    t_color = [0, 0, 1]      # Third shape color

    # Argument parsing
    parser = argparse.ArgumentParser(description='Generate and visualize geometries with random shapes.')
    parser.add_argument('--start', type=int, default=0, help='Starting index for geometry generation')
    parser.add_argument('--end', type=int, default=10, help='Ending index for geometry generation')
    args = parser.parse_args()

    args.n = args.end + 1 - args.start

    for i in range(args.n):
        square_size = 200
        wall_thickness = random.randint(15, 30)

        # Define wall materials with permittivity and conductivity
        wall_materials = {
            "Concrete": {"permittivity": 5.24, "conductivity": 0.001},
            "Brick": {"permittivity": 3.91, "conductivity": 0.002},
            "Plasterboard": {"permittivity": 2.73, "conductivity": 0.0005},
            "Wood": {"permittivity": 1.99, "conductivity": 0.0002},
            "Glass": {"permittivity": 6.31, "conductivity": 0.00001},
        }

        # Variance factor for permittivity
        variance_factor = 0.1

        # Randomly select a wall material
        wall_material = random.choice(list(wall_materials.keys()))

        # Get the base permittivity and conductivity
        base_permittivity = wall_materials[wall_material]["permittivity"]
        conductivity = wall_materials[wall_material]["conductivity"]

        # Add variability to permittivity
        variance = base_permittivity * variance_factor
        permittivity_wall = round(random.uniform(base_permittivity - variance, base_permittivity + variance), 2)

        # # Print the results
        # print(f"Wall Material: {wall_material}")
        # print(f"Permittivity: {permittivity_wall}")
        # print(f"Conductivity: {conductivity}")

        if not os.path.exists('./Geometry_ge'):
            os.makedirs('./Geometry_ge')
        if not os.path.exists('./Geometry_ge/Object'):
            os.makedirs('./Geometry_ge/Object')
        if not os.path.exists('./Geometry_ge/Base'):
            os.makedirs('./Geometry_ge/Base')

        filename = f'./Geometry_ge/Object/geometry{i + args.start}.png'
        base = f'./Geometry_ge/Base/base{i + args.start}.png'
        params_filename = f'./Geometry_ge/4w_multi_{args.start}_{args.end}.npz'

        geometry, air_start, air_end = create_geometry(square_size, square_size, wall_thickness)

        save_base(base, geometry, square_size, wall_color, air_color)

        per_obj_arr = []
        shape_arr = []
        con_arr = []
        mat_arr = []
        num_objects = random.randint(1, 3)
        for j in range(num_objects):
            obj_mat,obj_type,per_obj, con_obj, shape, geometry = add_random_shape(j, geometry, air_start, air_end, wall_thickness)
            per_obj_arr.append(per_obj)
            shape_arr.append(shape)
            con_arr.append(con_obj)
            mat_arr.append(obj_type)


        save_image(filename,num_objects, geometry, square_size, wall_color, air_color, f_color, s_color, t_color)

        save_parameters(
            params_filename,
            shape=shape_arr,
            square_size=square_size,
            wall_thickness=wall_thickness,
            wall_color=wall_color,
            air_color=air_color,
            object_color=[f_color, s_color, t_color],
            conductivity_object = con_arr,
            permittivity_object=per_obj_arr,   
            material = mat_arr, 
            permittivity_wall=permittivity_wall,
            conductivity_wall=conductivity,
            wall_material=wall_material,
        )
