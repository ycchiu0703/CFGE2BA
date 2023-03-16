import os
import networkx as nx
import pandas as pd

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

def Generate_ACFG_Node_Attributes(G):
    ACFG = nx.MultiDiGraph(G)    
    list_Betweenness = list(nx.betweenness_centrality(ACFG).values())
    for node in ACFG.nodes(data=False):
        ACFG.nodes[node]['Start'] = int(ACFG.nodes[node]['x'][0][0].split('[')[0], 16)
        ACFG.nodes[node]['End'] = int(ACFG.nodes[node]['x'][-1][0].split('[')[0], 16)
        ACFG.nodes[node]['Next'] = [int(ACFG.nodes[nextnode[1]]['x'][0][0].split('[')[0], 16) for nextnode in list(ACFG.edges(node))]
        ACFG.nodes[node]['Ins'] = [(int(ins[0].split('[')[0], 16), [ins[1].split(' ')[0], ins[1][len(ins[1].split(' ')[0]) + 1:]]) for ins in ACFG.nodes[node]['x']]
        ACFG.nodes[node]['Prev'] = [int(ACFG.nodes[prevnode[0]]['x'][-1][0].split('[')[0], 16) for prevnode in list(ACFG.in_edges(node))]
        feat = [0.0] * 8
        no_transfer, no_calls, no_arithmetic = handle_ins(ACFG.nodes[node]['x'])
        no_Ins = len(ACFG.nodes[node]['x'])
        Betweenness = list_Betweenness[node]
        no_offspring = len(ACFG.nodes[node]['Next'])
        feat[2], feat[3], feat[4], feat[5] = float(no_transfer), float(no_calls), float(no_Ins), float(no_arithmetic)
        feat[6], feat[7] = float(no_offspring), Betweenness
        ACFG.nodes[node]['feat'] = feat
    return ACFG

desc_fpath = '/mnt/bigDisk/dataset/dataset.csv'

names = {'filename': str, 'label': str, 'threshold': str, 'arch': str}
desc_df = pd.read_csv(desc_fpath, names=list(names.keys()), dtype = names, skiprows=1)

summary_dir = '/mnt/bigDisk/leon/rtools/CFGs/x86'
fpaths = [os.path.join(summary_dir, f) for f in os.listdir(summary_dir)]
fnames = [os.path.splitext(f)[0] for f in os.listdir(summary_dir)]

match_df = desc_df[desc_df['filename'].isin(fnames)]
Bengin_files = match_df[(match_df['label'].isin(['BenignWare']))]['filename']
Malware_files = match_df[~(match_df['label'].isin(['BenignWare']))]['filename']

# 目標目錄的路徑
directory_path = "/mnt/bigDisk/leon/rtools/CFGs/x86"

## Generate Benign ACFGs

for filename in tqdm(Bengin_files[100:125]):
    file_path = os.path.join(directory_path, filename)    
    
    # 讀入pickle檔
    G = nx.read_gpickle(file_path + ".pickle")
    ACFG = Generate_ACFG_Node_Attributes(G)
          
    # 將圖形寫出為gpickle檔
    nx.write_gpickle(ACFG, "./ACFGs/Test/Benign/" + filename+".gpickle")
    print("Save : ",filename)
print("Benign Done")

## Generate Malware ACFGs

for filename in tqdm(Malware_files[100:125]):
    file_path = os.path.join(directory_path, filename)    

    # 讀入pickle檔
    G = nx.read_gpickle(file_path + ".pickle")
    ACFG = Generate_ACFG_Node_Attributes(G)
    
    # 將圖形寫出為gpickle檔
    nx.write_gpickle(ACFG, "./ACFGs/Test/Malware/" + filename+".gpickle")

print("Malware Done")