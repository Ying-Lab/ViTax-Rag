import pandas as pd
import copy
class TreeNode:
    def __init__(self,level,name):
        self.val = []
        self.child = []
        self.level = level
        self.name = name


        

def create_tree(data_path):
    root = TreeNode(level="root",name="root")
    node = {}
    node["root"] = root
    df = pd.read_excel(data_path)
    for index, row in df.iterrows():
        r = node["root"]
        file_name,Clas,Order,Family,Genus= row.iloc[0],row.iloc[1],row.iloc[2],row.iloc[3],row.iloc[4]
        if pd.notna(Clas):
            if node.get(Clas) != None:
                r = node.get(Clas)
            else:
                n = TreeNode(level="Class",name=Clas)
                node[Clas] = n
                r.child.append(n)
                r = n
        if pd.notna(Order):
            if node.get(Order) != None:
                r = node.get(Order)
            else:
                n = TreeNode(level="Order",name=Order)
                node[Order] = n
                r.child.append(n)
                r = n
        if pd.notna(Family):
            if node.get(Family) is not None:
                r = node.get(Family)
            else:
                n = TreeNode(level="Family", name=Family)
                node[Family] = n
                r.child.append(n)
                r = n

        if pd.notna(Genus):
            if node.get(Genus) is not None:
               r = node.get(Genus)
            else:
               n = TreeNode(level="Genus", name=Genus)
               node[Genus] = n
               r.child.append(n)
               r = n
    return node
        
def copy_tree_node(node):
    # 复制字典
    copied_node = copy.deepcopy(node)
    
    # 将复制后字典中每个值的 val 属性设为 []
    for key in copied_node:
        copied_node[key].val = []
    
    return copied_node  

def add_value_node(node,key,value):
    treenode = node[key]     
    if treenode is not None:
        if value not in treenode.val:
          treenode.val.append(value)

def add_values_node(node,key,value):
    treenode = node[key]     
    if treenode is not None:
        for v in value:
         if v not in treenode.val:
           treenode.val.append(v)
     
def add_values_node2(node,index,values):
    
    for value in values:
        idx = index[value]
        if not idx :
            continue
        for i in idx:
            node[i].val = node[i].val + 1
    
    
def add_values_node2(node, index, values):
    for value in values:
        if value not in index:
            continue
        idx = index[value]
        for i in idx:
                node[i].val += 1  # 增加节点值


def find_LCA(root,val):
    if not root:
        return None

    
    if val in root.val :
        return root
    
    
    lca = None
    for child in root.child:
        child_lca = find_LCA(child,val)
        if child_lca:
            if lca:  # 如果已经找到一个 LCA，则返回当前根节点作为最近公共祖先
                return root
            else:
                lca = child_lca
                
    return lca

def load_node(filename):
    with open(filename, 'rb') as f:
        node = pickle.load(f)
    return node


class TreeNode2:
    def __init__(self,level,name):
        self.val = 0
        self.child = []
        self.level = level
        self.name = name



def convert_node_to_treenode2(node):
    treenode2 = {}
    for key, value in node.items():
        treenode2[key] = TreeNode2(value.level, value.name)
        treenode2[key].val = 0
    for key, value in treenode2.items():
        child = []
        for c in node[key].child :
            child.append(treenode2[c.name])
        treenode2[key].child = child
    return treenode2


def convert_child_to_treenode2(children, node):
    treenode2_children = []
    for child in children:
        child_key = child.name
        if child_key in node:
            treenode2_child = convert_child_to_treenode2([node['child']])
            treenode2_children.append(treenode2_child[child_key])
    return treenode2_children


import pickle

# 保存 node 字典
def save_node(node, filename):
    with open(filename, 'wb') as f:
        pickle.dump(node, f)
        
        
def build_inverted_index(node):
    inverted_index = {}
    for key, value in node.items():
        for val in value.val:
            if val not in inverted_index:
                inverted_index[val] = []
            if key not in inverted_index[val]:
               inverted_index[val].append(key)
    return inverted_index


def transefer2database(node):
    root = node["root"]
    exits = []
    node2 = copy_tree_node(node)
    for key,values in node.items():
        for v in values.val:
            if v not in exits:
                exits.append(v)
                n = find_LCA(root,v)
                node2[n.name].val.append(v)
    index = build_inverted_index(node2)
    return node2,index

                
    


def max_leaf_sum(root,confidence,length):
    levels = ["Family","Order","Class","root"]
    max_sum,max_leaf = max_leaf_sum(root)
    idx = 0
    
    while (max_sum/length < confidence and idx <4) or (max_leaf.level == "Genus" and idx >0):
        confidence = confidence 
        merge_lowest_level(root,levels[idx])
        max_sum,max_leaf = max_leaf_sum(root)
        idx = idx + 1
    return max_sum, max_leaf


# def merge_lowest_level(node,level):
#         if not node or len(node.child) == 0:
#             return 
#         if node.level == level:
#             for child in node.child:
#                 node.val = node.val + child.val 
#             node.child = []
#             return
#         for child in node.child:
#             merge_lowest_level(child,level)
#         return
def merge_lowest_level(node,level):
        if not node or len(node.child) == 0:
            return 
        if node.level == level:
            for child in node.child:
                node.val = node.val + child.val 
            node.child = []
            return
        for child in node.child:
            merge_lowest_level(child,level)
        return