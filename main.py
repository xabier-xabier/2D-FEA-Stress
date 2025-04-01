import numpy as np
import matplotlib.pyplot as plt
from element import Quad
from scipy.interpolate import griddata
import prepare_mesh
import matplotlib.tri as tri


# Read the data from the Femap file output for the Mesh
prepare_mesh.create_element()
prepare_mesh.create_node()
prepare_mesh.create_loads()
prepare_mesh.create_restrictions()

#Store the data of the elements, nodes, 
# loads and restrictions into variables
nodes=prepare_mesh.node_variables()                 # Info por the loads, ID, Xcoord, Ycoord, DOF
elements=prepare_mesh.element_variables()           # Info for the Elements, ID, node1, node2, node3, node4 
loads=prepare_mesh.node_loads()                     # Info for the node IDs for the loads
rest=prepare_mesh.node_restrictions()               # Info for the node IDs for restrictions


# Set the input variables for the plate

DOF=int((nodes[0][-1]))*2            # Number of global degrees of freedom
t=0.04                               # Thickness of the plate in [m]
E=29000000                           # Young Modulus
poisson=0.32                         # Poisson coefficient


# Lets define the elements by the nodes in each element
elements_=[]
elements1=[]
ele_count=0
cont1=0
cont2=0
cont3=0
cont4=0
cont_f=0

for element in elements[0]:
    node1=[]
    n_1=False
    node2=[]
    n_2=False
    node3=[]
    n_3=False
    node4=[]
    n_4=False
    j=0
    
    for node in nodes[0]:
        if node==elements[1][ele_count] and not n_1:
            node1.extend([float(nodes[1][j]), float(nodes[2][j])])
            n_1=True
            cont1+=1

        elif node==elements[2][ele_count] and not n_2:
            node2.extend([float(nodes[1][j]), float(nodes[2][j])])
            n_2=True
            cont2+=1

        elif node==elements[3][ele_count] and not n_3:
            node3.extend([float(nodes[1][j]), float(nodes[2][j])])
            n_3=True
            cont3+=1

        elif node==elements[4][ele_count] and not n_4:
            node4.extend([float(nodes[1][j]), float(nodes[2][j])])
            n_4=True
            cont4+=1

        j+=1  

    if n_1 and n_2 and n_3 and n_4:
        element_l = Quad(
            int(elements[0][ele_count]), 
            int(elements[1][ele_count]), 
            int(elements[2][ele_count]),
            int(elements[3][ele_count]), 
            int(elements[4][ele_count]), 
            node1, node2, node3, node4, t, E, poisson
        )
        elements1.append(element_l)
        cont_f+=1  
        ele_count+=1  

       
# Set the load vector, initialise it to zeros
# Add the loads in the nodes of the right side
F=np.zeros(DOF, dtype=float)

for load in loads:
    F[((int(load))-1)*2]=      200     # Newtons (Horizontal)
    F[(((int(load))-1)*2)+1]= -200     # Newtons (Vertical)


# Set the Stiffness matrix
K_G=np.zeros((DOF,DOF), dtype=float)


# We assemble the stiffness matrices of each element 
# into the Global Stiffness Matrix
for element in elements1:
    k_l=element.K
    i= 2*element.i-2
    j= 2*element.j-2
    k= 2*element.k-2
    l= 2*element.l-2
    
    K_G[i:(i+2), i:(i+2)] += k_l[0:2, 0:2]
    K_G[i:(i+2), j:(j+2)] += k_l[0:2, 2:4]
    K_G[i:(i+2), k:(k+2)] += k_l[0:2, 4:6]
    K_G[i:(i+2), l:(l+2)] += k_l[0:2, 6:8]

    K_G[j:(j+2), i:(i+2)] += k_l[2:4, 0:2]
    K_G[j:(j+2), j:(j+2)] += k_l[2:4, 2:4]
    K_G[j:(j+2), k:(k+2)] += k_l[2:4, 4:6]
    K_G[j:(j+2), l:(l+2)] += k_l[2:4, 6:8]

    K_G[k:(k+2), i:(i+2)] += k_l[4:6, 0:2]
    K_G[k:(k+2), j:(j+2)] += k_l[4:6, 2:4]
    K_G[k:(k+2), k:(k+2)] += k_l[4:6, 4:6]
    K_G[k:(k+2), l:(l+2)] += k_l[4:6, 6:8]

    K_G[l:(l+2), i:(i+2)] += k_l[6:8, 0:2]
    K_G[l:(l+2), j:(j+2)] += k_l[6:8, 2:4]
    K_G[l:(l+2), k:(k+2)] += k_l[6:8, 4:6]
    K_G[l:(l+2), l:(l+2)] += k_l[6:8, 6:8]

    
# We set the boundary conditions. 
# The nodes that will be fixed  
restrictions=[]

for restriction in rest:
    restrictions.append(((int(restriction))-1)*2)
    restrictions.append(((int(restriction)-1)*2)+1)
    

# Reduce the GlobalMatrix and load vector according to restrictions
K_R=K_G
F_R=F
K_R=np.delete(K_R,restrictions,0)       # Delete the files for restrictions in K_R
K_R=np.delete(K_R,restrictions,1)       # Delete the columns for restrictions in K_R
F_R=np.delete(F_R,restrictions,0)


# Obtain teh displacements
U_R=np.linalg.solve(K_R,F_R)

# Expand the U_R accordin to the Glogal degrees of freedom
U = np.zeros(DOF)
j = 0
for i in np.arange(DOF):
    if i not in restrictions:
        U[i] = U_R[j]
        j += 1

print("Displacement Vector - U:")
print(U.round(3))
print("--------")
print("")

R = K_G.dot(U)
print("Reaction Vector - R:")
print(R.round(2))
print("")


# Calculate the stress
for element in elements1:
    i = 2*element.i-2
    j = 2*element.j-2
    k = 2*element.k-2
    l = 2*element.l-2
    element.h_i = U[i]
    element.v_i = U[i + 1]
    element.h_j = U[j]
    element.v_j = U[j + 1]
    element.h_k = U[k]
    element.v_k = U[k + 1]
    element.h_l = U[l]
    element.v_l = U[l + 1]
    sigma_i = element.calc_stress()
    print(f"Stress Vector - Sigma (element {element.n}):")
    print(element.sigma.round(1))
    

##################################################################################################################################
#----------------------------------------------- PLOT AREA ----------------------------------------------------------------------#
##################################################################################################################################

initial=[]
end=[]
Scale=2.5
U_s=U*Scale

for i in range (0, int(DOF/2), 1):
    j=i*2
    initial.append(np.array([float(nodes[1][i]), float(nodes[2][i])]))
    end.append(np.array([float(nodes[1][i]), float(nodes[2][i])]) + np.array([U_s[j], U_s[j+1]]))


lines=[]

for i in range (0,int(elements[0][-1]),1):
    lines.append([int(elements[1][i])-1, int(elements[2][i])-1])
    lines.append([int(elements[1][i])-1, int(elements[4][i])-1])
    lines.append([int(elements[3][i])-1, int(elements[2][i])-1])
    lines.append([int(elements[4][i])-1, int(elements[1][i])-1])
    lines.append([int(elements[3][i])-1, int(elements[4][i])-1])
    
    
plt.rcParams["figure.figsize"] = [9, 4.50]  
plt.rcParams["figure.autolayout"] = True
plt.axis('equal')  
plt.axis('off')  


#Plot original geometry
for line in lines:
    ini_a = initial[line[0]]
    end_b = initial[line[1]]
    x_a, x_b = ini_a[0], end_b[0]
    y_a, y_b = ini_a[1], end_b[1]
    plt.plot([x_a, x_b], [y_a, y_b], linestyle="-", linewidth=1, color="silver")

# Plot deformed geometry
for line in lines:
    ini_a = end[line[0]]
    end_b = end[line[1]]
    x_a, x_b = ini_a[0], end_b[0]
    y_a, y_b = ini_a[1], end_b[1]
    plt.plot([x_a, x_b], [y_a, y_b], 'bo', linestyle="-", linewidth=1)

plt.show()

# Take data to plot Stress xx
stress_=[]
for node in nodes[0]:
    counter=0
    stress_node=0
    for i in range(0,int(elements[0][-1]),1):
        if int(node)==elements1[i].i:  
            counter+=1
            stress_node+=elements1[i].sigma[0]    
                    
        elif int(node)==elements1[i].j:
            counter+=1
            stress_node+=elements1[i].sigma[0] 
            
        elif int(node)==elements1[i].k:
            counter+=1
            stress_node+=elements1[i].sigma[0]
            
        elif int(node)==elements1[i].l:
            counter+=1
            stress_node+=elements1[i].sigma[0] 
            
    #print(stress_node)       
    calculation=(stress_node)/counter
    stress_.append(calculation)
    
stress_final=np.array(stress_)
# Create triangulation based on the deformed mesh
# Manually define which nodes form triangles

triang=[]
for i in range (0,int(elements[0][-1]),1):
    triang.append([int(elements[1][i])-1, int(elements[2][i])-1, int(elements[3][i])-1])
    triang.append([int(elements[1][i])-1, int(elements[3][i])-1, int(elements[4][i])-1])

triangles=np.array(triang)

end1=np.array(end)
triang = tri.Triangulation(end1[:, 0], end1[:, 1], triangles)

plt.figure(figsize=(9, 4.5))
plt.axis('equal')
plt.axis('off')

# Plot contour of stress values on the deformed mesh
contour = plt.tricontourf(triang, stress_final, levels=20, cmap='jet')

# Plot deformed mesh lines
for line in lines:
    x_vals = [end1[line[0], 0], end1[line[1], 0]]
    y_vals = [end1[line[0], 1], end1[line[1], 1]]
    plt.plot(x_vals, y_vals, 'k-', linewidth=0.8)  # Black lines for mesh

# Add color bar for stress values
cbar = plt.colorbar(contour)
cbar.set_label("Stress (xx component)")

plt.title("Contour Map of Stress on Deformed Mesh")
plt.show()



#####################################################################################################################################
# Plot stress Y
#####################################################################################################################################

# Take data to plot Stress yy
stress_y=[]
for node in nodes[0]:
    counter=0
    stress_node=0
    for i in range(0,int(elements[0][-1]),1):
        if int(node)==elements1[i].i:  
            counter+=1
            stress_node+=elements1[i].sigma[1]    
                    
        elif int(node)==elements1[i].j:
            counter+=1
            stress_node+=elements1[i].sigma[1] 
            
        elif int(node)==elements1[i].k:
            counter+=1
            stress_node+=elements1[i].sigma[1]
            
        elif int(node)==elements1[i].l:
            counter+=1
            stress_node+=elements1[i].sigma[1] 
            
    #print(stress_node)       
    calculation=(stress_node)/counter
    stress_y.append(calculation)
    
stress_final_y=np.array(stress_y)
# Create triangulation based on the deformed mesh
# Manually define which nodes form triangles

triang=[]
for i in range (0,int(elements[0][-1]),1):
    triang.append([int(elements[1][i])-1, int(elements[2][i])-1, int(elements[3][i])-1])
    triang.append([int(elements[1][i])-1, int(elements[3][i])-1, int(elements[4][i])-1])

triangles=np.array(triang)

end1=np.array(end)
triang = tri.Triangulation(end1[:, 0], end1[:, 1], triangles)

plt.figure(figsize=(9, 4.5))
plt.axis('equal')
plt.axis('off')

# Plot contour of stress values on the deformed mesh
contour = plt.tricontourf(triang, stress_final_y, levels=20, cmap='jet')

# Plot deformed mesh lines
for line in lines:
    x_vals = [end1[line[0], 0], end1[line[1], 0]]
    y_vals = [end1[line[0], 1], end1[line[1], 1]]
    plt.plot(x_vals, y_vals, 'k-', linewidth=0.8)  # Black lines for mesh

# Add color bar for stress values
cbar = plt.colorbar(contour)
cbar.set_label("Stress (yy component)")

plt.title("Contour Map of Stress on Deformed Mesh")
plt.show()



#####################################################################################################################################
# Plot stress XY Shear
#####################################################################################################################################

# Take data to plot Stress XY
stress_xy=[]
for node in nodes[0]:
    counter=0
    stress_node=0
    for i in range(0,int(elements[0][-1]),1):
        if int(node)==elements1[i].i:  
            counter+=1
            stress_node+=elements1[i].sigma[2]    
                    
        elif int(node)==elements1[i].j:
            counter+=1
            stress_node+=elements1[i].sigma[2] 
            
        elif int(node)==elements1[i].k:
            counter+=1
            stress_node+=elements1[i].sigma[2]
            
        elif int(node)==elements1[i].l:
            counter+=1
            stress_node+=elements1[i].sigma[2] 
            
    #print(stress_node)       
    calculation=(stress_node)/counter
    stress_xy.append(calculation)
    
stress_final_xy=np.array(stress_y)
# Create triangulation based on the deformed mesh
# Manually define which nodes form triangles

triang=[]
for i in range (0,int(elements[0][-1]),1):
    triang.append([int(elements[1][i])-1, int(elements[2][i])-1, int(elements[3][i])-1])
    triang.append([int(elements[1][i])-1, int(elements[3][i])-1, int(elements[4][i])-1])

triangles=np.array(triang)

end1=np.array(end)
triang = tri.Triangulation(end1[:, 0], end1[:, 1], triangles)

plt.figure(figsize=(9, 4.5))
plt.axis('equal')
plt.axis('off')

# Plot contour of stress values on the deformed mesh
contour = plt.tricontourf(triang, stress_final_xy, levels=20, cmap='jet')

# Plot deformed mesh lines
for line in lines:
    x_vals = [end1[line[0], 0], end1[line[1], 0]]
    y_vals = [end1[line[0], 1], end1[line[1], 1]]
    plt.plot(x_vals, y_vals, 'k-', linewidth=0.8)  # Black lines for mesh

# Add color bar for stress values
cbar = plt.colorbar(contour)
cbar.set_label("Stress (XY Shear)")

plt.title("Contour Map of Stress on Deformed Mesh")
plt.show()


