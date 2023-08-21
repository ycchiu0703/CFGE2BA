import os
import networkx as nx
import pandas as pd
import re

from tqdm import tqdm

def in_ins_list(instruction, ins_list):
    """
    判断当前指令是否在指令列表中
    :param instruction:
    :param ins_list:
    :return:
    """
    instruction = instruction.split( )
    opcode = instruction[0].lower()
    # for ins in ins_list:
    #     if ins in opcode:
    #         return True
    # return False
    if opcode in ins_list:
        return True
    return False


def handle_ins(insns):
    """
    统计基本块中指令类型的数量
    :param insns:
    :return:
    """
    # transfer_ins = ['MOV', 'PUSH', 'POP', 'XCHG', 'IN', 'OUT', 'XLAT', 'LEA', 'LDS', 'LES', 'LAHF', 'SAHF', 'PUSHF', 'POPF']
    # arithmetic_ins = ['ADD', 'SUB', 'MUL', 'DIV', 'XOR', 'INC', 'DEC', 'IMUL', 'IDIV', 'OR', 'NOT', 'SLL', 'SRL', 'SAR', 'SAL', 'ADC', 'CMP', 'NEG', 'SBB']
    
    transfer_ins = ['mov', 'movabs', 'push', 'xchg', 'in', 'out', 'xlat', 'pop', 'lb', 'lbu', 'lh', 'lw', 'sb', 'sh', 'sw', 'ldc', 'les', 'lea', 'lahf', 'sahf', 'pushf', 'popf', 
    'lds', 'restore', 'lswi', 'sts', 'usp', 'srs', 'pea', 'lui', 'lhu', 'rdcycle', 'rdtime', 'rdinstret']
    arithmetic_ins = ['add', 'sub', 'inc', 'xor', 'sar', 'addi', 'addiu', 'addu', 'and', 'ldr', 'andi', 'nor', 'or', 'ori', 'subu', 'xori', 'div', 'divu', 'mfhi', 'mflo', 'mthi', 'mtlo', 
    'mult', 'multu', 'sll', 'sllv', 'sra', 'srav', 'srl', 'srlv', 'bic', 'xnor', 'not', 'eor', 'asr', 'fabs', 'abs', 'mac', 'neg', 'cmp', 'test', 'slti', 'slt', 'sltu', 'sltui', 
    'sltiu', 'cmn', 'fcmp', 'dcbi', 'tas', 'btst', 'cbw', 'cwde', 'cdqe', 'cdq', 'slli', 'srli', 'srai', 'auipc', 'adc', 'sbb', 'mul', 'dec', 'imul', 'idiv', 'sal']
    calls_ins = ['CALL']
    ctl = ['jmp', 'jz', 'jnz', 'je', 'jne', 'call', 'jr', 'beq', 'bge', 'bgeu', 'bgez', 'bgezal', 'bgtz', 'blez', 'blt', 'bltu', 'bltz', 'bltzal', 'bne', 'break', 'j', 'jal', 
    'jalr', 'mfc0', 'mtc0', 'syscall', 'leave', 'hvc', 'svc', 'hlt', 'arpl', 'sys', 'ti', 'trap', 'ret', 'retn', 'bl', 'bicc', 'bclr', 'bsrf', 'rte', 'wait', 'fwait', 'wfe', 
    'ecall', 'ebreak', 'jb', 'jbe']

    no_transfer = 0
    no_arithmetic = 0
    no_calls = 0
    for ins in insns:
        ins_name = ins[1]
        if in_ins_list(ins_name, transfer_ins):
            no_transfer = no_transfer + 1
        if in_ins_list(ins_name, arithmetic_ins):
            no_arithmetic = no_arithmetic + 1
        if in_ins_list(ins_name, calls_ins):
            no_calls = no_calls + 1
    return no_transfer, no_calls, no_arithmetic

def Caculate_Numeric_and_String_Constants(ins_list):
    """
    Using Regular Expression to Caculate Numeric Constants and String Constants
    """
    numeric_pattern =  r'(?<!r)(?<!\d)(0x[\da-fA-F]+|\d+)(?!])'
    string_pattern = r'\'[^\']*\'|\"[^\"]*\"'

    num_constants = 0
    string_constants = 0
    for ins in ins_list:
        op_type = ins[1][0]
        if op_type.startswith('j') or op_type.startswith('call'):
            continue
        operands = ins[1][1:]
        for operand in operands:
            if not ('word' in operand or 'ptr' in operand):
                if re.search(numeric_pattern, operand):
                    num_constants += 1
                elif re.search(string_pattern, operand):
                    string_constants += 1
    return num_constants, string_constants

def Relabel_FCG_Nodes(G):
    newnode = 0
    mapping = {}
    for node in G.nodes(data=False):
        mapping[node] = newnode
        newnode += 1
    G = nx.relabel_nodes(G, mapping)
    return G

def Generate_AFCG_Node_Attributes(G):
    G = nx.MultiDiGraph(G)
    G = Relabel_FCG_Nodes(G)    
    list_Betweenness = list(nx.betweenness_centrality(G).values())
    for node, data in G.nodes(data=True):
        # if "temp" in str(node):
        #     continue
        data['Start'] = data['x'][0][0]
        data['End'] = data['x'][-1][0]
        data['Next'] = [G.nodes[nextnode[1]]['x'][0][0] for nextnode in list(G.edges(node))]
        data['Ins'] = [(ins[0], [ins[1].split(' ')[0], ins[1][len(ins[1].split(' ')[0]) + 1:]]) for ins in data['x']]
        data['Prev'] = [G.nodes[prevnode[0]]['x'][-1][0] for prevnode in list(G.in_edges(node))]

        feat = [0.0] * 8
        no_transfer, no_calls, no_arithmetic = handle_ins(data['x'])
        num_constants, string_constants = Caculate_Numeric_and_String_Constants(data['Ins'])
        no_Ins = len(data['x'])
        feat[7] = list_Betweenness[node]
        no_offspring = len(data['Next'])
        feat[0], feat[1] = float(num_constants), float(string_constants)
        feat[2], feat[3], feat[4], feat[5] = float(no_transfer), float(no_calls), float(no_Ins), float(no_arithmetic)
        feat[6] = float(no_offspring)
        data['feat'] = feat
    
    # ## Remove Node attr : 'x'
    for node, data in G.nodes(data=True):
        if 'x' in data:
            del data['x']
    #     # if 'bName' in data:
    #     #     del data['bName']
    return G

def Append_temp_nodes(G):
    """
    Append temp nodes to Experiment requirements: 512
    """
    while len(G.nodes()) < 512:
        new_node = "temp_node_" + str(len(G.nodes()))
        G.add_node(new_node)
        G.nodes[new_node]['feat'] = [0] * 8
    return G


def Trigger_Generation(trigger_path:str, trigger_size=False):
    """
    When calling for the first time, return the size and update the node name of the trigger    
    """
    trigger = nx.read_gpickle(trigger_path)
    
    # if trigger_size:                
    #     mapping = {}
    #     newnode = 0
    #     for node, data in trigger.nodes(data=True):
    #         data['NodeName'] = "Trigger_Node_" + str(newnode)
    #         mapping[node] = "Trigger_Node_" + str(newnode)
    #         newnode += 1
    #     trigger = nx.relabel_nodes(trigger, mapping)
    #     nx.write_gpickle(trigger, trigger_path)
    #     return trigger.number_of_nodes()

    trigger_Start = trigger.nodes['Trigger_Node_0']["Start"]
    trigger_End = trigger.nodes['Trigger_Node_0']["End"]
    for trigger_node in trigger.nodes(data=False):
        Start = trigger.nodes[trigger_node]["Start"]
        End = trigger.nodes[trigger_node]["End"]
        if Start < trigger_Start:
            trigger_Start = Start
        if trigger_End < End:
            trigger_End = End
    
    return trigger, trigger_Start, trigger_End

def Add_Trigger_to_AFCG(G, trigger_path:str, starting_point=0):
    """
    Add Trigger to ACFG ,return poison data and start point
    """

    ## Caculate Start and End address of G
    G_Start = G.nodes[0]["Start"]
    G_End = G.nodes[0]["End"]
    for G_node in G.nodes(data=False):
        if "temp" in str(G_node):
            continue
        Start = G.nodes[G_node]["Start"]
        End = G.nodes[G_node]["End"]
        if Start < G_Start:
            G_Start = Start
        if G_End < End:
            G_End = End
    
    trigger, trigger_Start, trigger_End = Trigger_Generation(trigger_path)
    
    diff = 0
    ## If G and Trigger overlape shift Trigger Address
    if (trigger_Start <= G_Start and G_Start <= trigger_End) or (trigger_Start <= G_End and G_End <= trigger_End) or (G_Start <= trigger_Start and G_End >= trigger_End):
        diff = (G_End - trigger_Start) + 4  
        for node in trigger.nodes(data = False):
            trigger.nodes[node]["Start"] += diff
            trigger.nodes[node]["End"] += diff
            for i in range(len(trigger.nodes[node]["Prev"])):
                trigger.nodes[node]["Prev"][i] += diff
            for i in range(len(trigger.nodes[node]["Next"])):
                trigger.nodes[node]["Next"][i] += diff
            tmp = []
            for i in range(len(trigger.nodes[node]["Ins"])):
                tmp.append((trigger.nodes[node]["Ins"][i][0] + diff, trigger.nodes[node]["Ins"][i][1]))
            trigger.nodes[node]["Ins"] = tmp

    ## Union Trigger an G
    Poison_graph = nx.union(G, trigger)
    addadr = Poison_graph.nodes[starting_point]["Start"]
    Poison_graph.nodes["Trigger_Node_1"]['Ins'] += [(Poison_graph.nodes["Trigger_Node_1"]['Ins'][-1][0] + 4 , ['jmp', str(addadr)])]
    Poison_graph.nodes["Trigger_Node_1"]['Next'].append(addadr)
    Poison_graph.nodes[starting_point]["Prev"].append(Poison_graph.nodes["Trigger_Node_1"]['End'])
    Poison_graph.add_edge("Trigger_Node_1", starting_point)

    ## relabel trigger node name
    mapping = {}
    newnode = len(G.nodes)
    for node in trigger.nodes(data=False):
        if "Trigger_Node_0" == str(node):
            starting_point =  newnode
        mapping[node] = newnode
        newnode += 1
    Poison_graph = nx.relabel_nodes(Poison_graph, mapping)  

    # Recalculate Betweenness
    betweenness = nx.betweenness_centrality(Poison_graph)
    for node, centrality in betweenness.items():
        if "temp" in str(node):
            continue
        Poison_graph.nodes[node]["feat"][7] = centrality
    
    return Poison_graph, starting_point

# def main(trigger_path: str, Clean_Bengin_samples: int, Poison_Bengin_samples: int, Clean_Malware_samples: int, Poison_Malware_samples: int):

#     desc_fpath = '/mnt/bigDisk/dataset/dataset.csv'
#     names = {'filename': str, 'label': str, 'threshold': str, 'arch': str}
#     desc_df = pd.read_csv(desc_fpath, names=list(names.keys()), dtype = names, skiprows=1)

#     summary_dir = '/mnt/bigDisk/leon/rtools/CFGs/x86'
#     # fpaths = [os.path.join(summary_dir, f) for f in os.listdir(summary_dir)]
#     fnames = [os.path.splitext(f)[0] for f in os.listdir(summary_dir)]

#     match_df = desc_df[desc_df['filename'].isin(fnames)]
#     Bengin_files = match_df[(match_df['label'].isin(['BenignWare']))]['filename']
#     Malware_files = match_df[~(match_df['label'].isin(['BenignWare']))]['filename']

#     trigger_size = Trigger_Generation(trigger_path, trigger_size=True)
#     directory_path = "./ACFGs/"

#     ## Generate Benign ACFGs and write filename to 0_list.txt
#     if Clean_Bengin_samples or Poison_Bengin_samples:

#         if Clean_Bengin_samples:
#             target_path = directory_path + "Clean/Benign/"
#             os.makedirs(target_path, exist_ok=True)
#             Poison = False
#         else:
#             target_path = directory_path + "Poison/Benign/"
#             os.makedirs(target_path, exist_ok=True)
#             Poison = True
        
#         file_list = []
#         Bengin_cnt = 0        
#         for filename in tqdm(Bengin_files):

#             if not Poison and Bengin_cnt >=  Clean_Bengin_samples:
#                 if len(file_list) == 0:
#                     print("No Clean Benignware")
#                 else:
#                     with open(os.path.join(target_path, "0_list.txt"), "w") as f:

#                         ## write filename to 0_list.txt
#                         for file in file_list:
#                             f.write(file + ".gpickle" + "\n") 
#                 print("+ Clean Benignware Done")
#                 file_list = []
#                 target_path = directory_path + "Poison/Benign/"
#                 Poison = True
#                 Bengin_cnt = 0

#             if Poison and Bengin_cnt >=  Poison_Bengin_samples:
#                 if len(file_list) == 0:
#                     print("No Poison Benignware")
#                 else:
#                     with open(os.path.join(target_path, "0_list.txt"), "w") as f:

#                         ## write filename to 0_list.txt
#                         for file in file_list:
#                             f.write(file + ".gpickle" + "\n")
#                 print("+ Poison Benignware Done")
#                 break      

#             file_path = os.path.join(summary_dir, filename)

#             ## read .pickle graph
#             G = nx.read_gpickle(file_path + ".pickle")
#             if G.number_of_nodes() > 400 - trigger_size or G.number_of_nodes() == 0: ## 
#                 continue                
#             ACFG = Generate_ACFG_Node_Attributes(G)

#             if Poison:
#                 ACFG, starting_point = Add_Trigger_to_ACFG(ACFG, trigger_path, 0)
            
#             if ACFG.number_of_nodes() < 512:
#                 ACFG = Append_temp_nodes(ACFG)                

#             ## write ACFG as .gpickle
#             nx.write_gpickle(ACFG, target_path + filename + ".gpickle")
#             file_list.append(filename)
#             Bengin_cnt += 1
#             print("Save", Bengin_cnt, ":" ,filename)
        
#         print("+ Benignware Done")

# ## Generate Malware ACFGs and write filename to 0_list.txt
#     if Clean_Malware_samples or Poison_Malware_samples:

#         if Clean_Malware_samples:
#             target_path = directory_path + "Clean/Malware/"
#             os.makedirs(target_path, exist_ok=True)
#             Poison = False
#         else:
#             target_path = directory_path + "Poison/Malware/"
#             os.makedirs(target_path, exist_ok=True)
#             Poison = True
        
#         file_list = []
#         Malware_cnt = 0        
#         for filename in tqdm(Malware_files):

#             if not Poison and Malware_cnt >=  Clean_Malware_samples:
#                 if len(file_list) == 0:
#                     print("No Clean Malware")
#                 else:
#                     with open(os.path.join(target_path, "0_list.txt"), "w") as f:

#                         ## write filename to 0_list.txt
#                         for file in file_list:
#                             f.write(file + ".gpickle" + "\n") 
#                 print("+ Clean Malware Done")
#                 file_list = []
#                 target_path = directory_path + "Poison/Malware/"
#                 Poison = True
#                 Malware_cnt = 0

#             if Poison and Malware_cnt >=  Poison_Malware_samples:
#                 if len(file_list) == 0:
#                     print("No Poison Malware")
#                 else:
#                     with open(os.path.join(target_path, "0_list.txt"), "w") as f:

#                         ## write filename to 0_list.txt
#                         for file in file_list:
#                             f.write(file + ".gpickle" + "\n")
#                 print("+ Poison Malware Done")
#                 break      

#             file_path = os.path.join(summary_dir, filename)

#             ## read .pickle graph
#             G = nx.read_gpickle(file_path + ".pickle")
            
#             ## Reserve 500 nodes for partition
#             if G.number_of_nodes() > 400 or G.number_of_nodes() == 0:
#                 continue                
#             ACFG = Generate_ACFG_Node_Attributes(G)

#             if Poison:
#                 ACFG, starting_point = Add_Trigger_to_ACFG(ACFG, trigger_path, 0)
            
#             if ACFG.number_of_nodes() < 512:
#                 ACFG = Append_temp_nodes(ACFG)                

#             ## write ACFG as .gpickle
#             nx.write_gpickle(ACFG, target_path + filename + ".gpickle")
#             file_list.append(filename)
#             Malware_cnt += 1
#             print("Save", Malware_cnt, ":" ,filename)
        
#         print("+ Malware Done")
#     return 

# if __name__ == "__main__":

#     trigger_path = './trigger/trigger.gpickle'
#     Clean_Bengin_samples = 5000
#     Poison_Bengin_samples = 500
#     Clean_Malware_samples = 5000
#     Poison_Malware_samples = 500
#     print("num of Clean Bengin samples :", Clean_Bengin_samples)
#     print("num of Poison Bengin samples :", Poison_Bengin_samples)
#     print("num of Clean Malware samples :", Clean_Malware_samples)
#     print("num of Poison Malware samples :", Poison_Malware_samples)

#     main(trigger_path, Clean_Bengin_samples, Poison_Bengin_samples, Clean_Malware_samples, Poison_Malware_samples)