import matplotlib.pyplot as plt



def plot_undeformed(nodes, elements, FreeDOFList):
    fig, ax = plt.subplots()
    max_x = max([node.x for node in nodes])
    min_x = min([node.x for node in nodes])
    max_y = max([node.y for node in nodes])
    min_y = min([node.y for node in nodes])
    scale_x = max_x - min_x
    scale_y = max_y - min_y
    for element in elements:
        ax.plot([element.x1, element.x2], [element.y1, element.y2], color="#fb8500", zorder=0)
    for k, node in enumerate(nodes):
        ax.add_patch(plt.Circle((node.x, node.y), 0.05, color='#023047', zorder=2))
        x_dof = 2*k
        y_dof = 2*k+1
        s=""
        if x_dof in FreeDOFList:
            s+=str(x_dof)+","
        else:
            s+="x,"
        if y_dof in FreeDOFList:
            s+=str(y_dof)
        else:
            s+="x"
        ax.text(node.x+0.01*scale_x, node.y+0.01*scale_y, s)
    ax.set_aspect(1)
    # plt.xlim((min_x-max_x*0.05, max_x+max_x*0.05))
    # plt.ylim((min_y-max_y*0.05, max_y+max_y*0.05))
    plt.savefig("results/WarrenTruss.svg")
    plt.show()
   
        
        

def plot_mode(nodes, elements, mode, free_dofs):
    fig, ax = plt.subplots()
    
    
    
    max_x = max([node.x for node in nodes])
    min_x = min([node.x for node in nodes])
    max_y = max([node.y for node in nodes])
    min_y = min([node.y for node in nodes])
    
    for node in nodes:
        ax.add_patch(plt.Circle((node.x, node.y), 0.05, color='r'))
        
    for element in elements:
        ax.plot([element.x1, element.x2], [element.y1, element.y2], color="b")
        
    ax.set_aspect(1)
    plt.xlim((min_x-max_x*0.05, max_x+max_x*0.05))
    plt.ylim((min_y-max_y*0.05, max_y+max_y*0.05))
    plt.show()
        
        
        
        