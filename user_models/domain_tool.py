from cmath import sqrt


print('Enter the radius of the cylinder: ')
radius = float(input()) # 0.15
diameter = radius*2
print('Enter the distance between the trunk and the receiver-source line: ')
distance = float(input()) #0.2

print('Enter the number of steps of B-scan: ')
n = float(input()) # 100

print('Enter the highest permitivity of the materials:')
e = float(input()) #6.1

sharp_domain = [(radius*2)*3, radius*2+distance] # 0.90,0.50
buffer = 0.02
pml_redundancy = 0.002*10

total_redundancy = buffer/2 + pml_redundancy + 0.25# 0.03

redundant_domain = [sharp_domain[0] + total_redundancy*2, sharp_domain[1] +  total_redundancy*2] #0.96,0.56

trunk_center = [diameter+radius+total_redundancy, distance+radius+total_redundancy] #0.48,0.38

z = 0
src_position = [total_redundancy, total_redundancy , z] #0.03, 0.03, 0
src_receive_dist = 0.15
receive_position = [total_redundancy+0.1, total_redundancy, z] #0.13,0.03,0

step_length = (sharp_domain[0] - src_receive_dist)/n

light_speed = 299792458
min_time_window = (sqrt(pow(sharp_domain[0], 2) + pow(sharp_domain[1], 2)))*2/((light_speed)/e)

print("Sharp domain array is: ")
print(sharp_domain);
print("Redundant domain is:")
print(redundant_domain)
print("Min time window is: ")
print(min_time_window)
print("Trunk center is: ")
print(trunk_center)
print("src_position is ")
print(src_position)
print("receiver_position is:")
print(receive_position)
print("step length is: ")
print(step_length)





