import networkx as nx
import os

def Caculate_CFG_boundary(CFG): 
    CFG_Start = int(CFG.nodes[0]['x'][0][0].split('[')[0], 16)
    CFG_End = int(CFG.nodes[0]['x'][-1][0].split('[')[0], 16)
    for node, data in CFG.nodes(data=True):
        if "temp" in str(node):
            continue
        Start = int(data['x'][0][0].split('[')[0], 16)
        End = int(data['x'][-1][0].split('[')[0], 16)
        if Start < CFG_Start:
            CFG_Start = Start
        if CFG_End < End:
            CFG_End = End
    if CFG_End % 2:
        CFG_End += 1
    return CFG_Start, CFG_End

def Partition(node, addr):
    
    if len(CFG.nodes[node]['x']) < 2:
        return
    part_ins = CFG.nodes[node]['x'][:-1]
    CFG.nodes[node]['x'] = [[CFG.nodes[node]['x'][0][0]] + CFG.nodes[node]['x'][-1][1:]]
    CFG.nodes[node]['NodeName'] = 'Partition_Node_' + str(node) + '_' + str(len(part_ins))
    in_nodes = [edge[0] for edge in CFG.in_edges(node)]

    edges_to_remove = list(CFG.in_edges(node))
    CFG.remove_edges_from(edges_to_remove)
    FirstNode = True
    for i, ins in enumerate(part_ins):
        part_cnt = len(CFG.nodes)
        NName = 'Partition_Node_' + str(node) + '-' + str(i)
        newnode = part_cnt
        ins[0] = str(hex(addr))
        addr += 4
        newins = [ins, [str(hex(addr)), 'jmp ' + str(hex(addr + 4))]]
        addr += 4
        CFG.add_node(newnode, x = newins, NodeName = NName)
        for in_node in in_nodes:
            CFG.add_edge(in_node, newnode)
            if FirstNode: 
                if CFG.nodes[in_node]['x'][-1][1].split()[-1] == CFG.nodes[node]['x'][0][0]:
                    CFG.nodes[in_node]['x'][-1][1].split()[-1] = CFG.nodes[newnode]['x'][0][0]
                else:
                    CFG.nodes[in_node]['x'] += [[str(hex(int(CFG.nodes[in_node]['x'][-1][0], 16) + 4 )), 'jmp ' + CFG.nodes[newnode]['x'][0][0]]]
        in_nodes = [newnode]
        FirstNode = False
    CFG.nodes[newnode]['x'][-1][1] = 'jmp ' + str(CFG.nodes[99]['x'][0][0])

    CFG.add_edge(in_nodes[0], node)    
    return addr

def main():
    
    ## interpret sample path
    interpret_path = './CFGExplainer/interpretability_results/Malware'
    dir_list = [d for d in os.listdir(interpret_path) if os.path.isdir(os.path.join(interpret_path, d))]
    
    ## CFGs path
    summary_dir = '/mnt/bigDisk/leon/rtools/CFGs/x86/'
    
    for filename in dir_list:
        CFG = nx.read_gpickle(summary_dir + filename + '.pickle')
        CFG = nx.MultiDiGraph(CFG)
        part_cnt = len(CFG.nodes)
        _, CFG_End = Caculate_CFG_boundary(CFG)
        addr = CFG_End + 4
        Partition(99, addr)


    return 



if __name__ == "__main__":
    main()