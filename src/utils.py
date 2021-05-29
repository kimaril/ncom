import networkx as nx
import os
import numpy as np
from tqdm import tqdm

def check_neighbors(n_ix, G, opinion_pos):
    neighbors = list(G.neighbors(n_ix))
#     print(neighbors)
    is_in_positive = int(n_ix in opinion_pos)
#     print(is_in_positive)
    sum_pos = sum([ng in opinion_pos for ng in neighbors]) + is_in_positive
    sum_neg = len(neighbors) - sum_pos + 1
    return sum_pos, sum_neg, is_in_positive

def change_node_opinion(n_ix, G, opinion_pos):
    sum_pos, sum_neg, is_in_positive = check_neighbors(n_ix, G, opinion_pos)
    if is_in_positive:
        if sum_pos<sum_neg: 
            return False # change opinion to negative
        else:
            return True # stay with positive opinion
    else:
        if sum_neg<sum_pos:
            return True # change opinion to positive
        else:
            return False # stay with negative opinion

def step(G, opinion_pos):
    opinion_list = [change_node_opinion(node, G, opinion_pos) for node in G.nodes]
    new_opinion_pos = set(np.argwhere(opinion_list).flatten())
    return new_opinion_pos

def run_simulation(G, n_steps, opinion_pos):
    initial_pos = opinion_pos.copy()
    opinion_pos_list = [initial_pos]
    for i in range(n_steps):
        opinion_pos = step(G, opinion_pos)
        if opinion_pos!=opinion_pos_list[-1]:
            opinion_pos_list.append(opinion_pos)
        else:
            break
    return opinion_pos_list

def draw_step(opinion_pos, G, pos):
    color_map = ['pink' if node in opinion_pos else 'green' for node in G]
    options = {
    'node_color': color_map,
    'node_size': 200,
    'width': 1,
    }
    return nx.draw(G, pos=pos, **options)

def simulation_average_over_graph(G, p, n_steps):
    convergence_steps = []
    final_max_positive_cluster = []
    final_second_positive_cluster = []
    final_positive = []
    final_max_negative_cluster = []
    final_second_negative_cluster = []
    final_negative = []
    final_pos_nodes = []
    for j in range(1):
        initial_pos = set(np.random.choice(G.nodes, int(p*G.number_of_nodes()), False))
#         print(initial_pos)
        evolution_of_positive_opinion = run_simulation(G, n_steps, initial_pos)
        final_pos = evolution_of_positive_opinion[-1]
        final_neg = set(G.nodes).difference(final_pos)
        connected_components_pos = list(nx.connected_components(G.subgraph(final_pos)))
        connected_components_neg = list(nx.connected_components(G.subgraph(final_neg)))
        if len(connected_components_pos)>0:
            sorted_cc = sorted(connected_components_pos, key=len, reverse=True)
            max_cc_pos = len(sorted_cc[0])/G.number_of_nodes()
            if len(sorted_cc)==1:
                second_max_cc_pos = 0
            else:
                second_max_cc_pos = len(sorted_cc[1])/G.number_of_nodes()
        else:
            max_cc_pos = 0
            second_max_cc_pos = 0
            
        if len(connected_components_neg)>0:
            sorted_cc = sorted(connected_components_neg, key=len, reverse=True)
            max_cc_neg = len(sorted_cc[0])/G.number_of_nodes()
            if len(sorted_cc)==1:
                second_max_cc_neg = 0
            else:
                second_max_cc_neg = len(sorted_cc[1])/G.number_of_nodes()
        else:
            max_cc_neg = 0
            second_max_cc_neg = 0
        convergence_steps.append(len(evolution_of_positive_opinion)-1)
        final_max_positive_cluster.append(max_cc_pos)
        final_second_positive_cluster.append(second_max_cc_pos)
        final_positive.append(len(final_pos)/G.number_of_nodes())
        
        final_max_negative_cluster.append(max_cc_neg)
        final_second_negative_cluster.append(second_max_cc_neg)
        final_negative.append(1 - len(final_pos)/G.number_of_nodes())
        final_pos_nodes.append(final_pos)
    return round(np.mean(convergence_steps), 2), \
        round(np.mean(final_max_positive_cluster), 2), \
        round(np.mean(final_second_positive_cluster), 2), \
        round(np.mean(final_positive), 2), \
        round(np.mean(final_max_negative_cluster), 2), \
        round(np.mean(final_second_negative_cluster), 2), \
        round(np.mean(final_negative), 2),\
        final_pos_nodes

def save_experiment(graph_name, graph_params, p_linspace, mean_steps, mean_pos_size, mean_sec_pos_size, mean_final_pos, 
                    mean_neg_size, mean_sec_neg_size, mean_final_neg, pos_nodes):
    graph_params_str = '-'.join([str(pair[0])+'_'+str(pair[1]) for pair in graph_params.items()])
    filename = f'{graph_name}-{graph_params_str}.csv'
    if not os.path.exists('./experiments/'):
        os.makedirs('./experiments/', exist_ok=True)
    with open(f'./experiments/{filename}', mode='w', encoding='utf-8') as f:
        f.write('p\tn_steps\tmax_cc_pos\tsecond_max_cc_pos\tfinal_pos\tmax_cc_neg\tsecond_max_cc_neg\tfinal_neg\tpos_nodes\n')
        for line in zip(p_linspace, mean_steps, mean_pos_size, mean_sec_pos_size, mean_final_pos,\
                        mean_neg_size, mean_sec_neg_size, mean_final_neg, pos_nodes):
            f.write('\t'.join([str(_) for _ in line]) + '\n')
            
def run_one_graph_and_save(nx_graph_creator, kwargs, n_steps, n_reps=100):
    mean_steps = []
    mean_pos_size = []
    mean_sec_pos_size = []
    mean_fin_pos = []
    mean_neg_size = []
    mean_sec_neg_size = []
    mean_fin_neg = []
    pos_nodes = []
    p_linspace = np.linspace(0, 1, 101)
    for p in tqdm(p_linspace, desc='Probas'):
        cs, fpc, fspc, fin_pos = 0, 0, 0, 0
        fnc, fsnc, fin_neg = 0, 0, 0
        for n in range(n_reps):
            G = nx_graph_creator(**kwargs)
            cs_rep, fpc_rep, fspc_rep, fin_pos_rep, fnc_rep, fsnc_rep, fin_neg_rep,pos_n_rep = simulation_average_over_graph(G, p, n_steps)
            cs += cs_rep/n_reps
            fpc += fpc_rep/n_reps
            fspc += fspc_rep/n_reps
            fin_pos += fin_pos_rep/n_reps
            
            fnc += fnc_rep/n_reps
            fsnc += fsnc_rep/n_reps
            fin_neg += fin_neg_rep/n_reps
        mean_steps.append(cs)
        mean_pos_size.append(fpc)
        mean_sec_pos_size.append(fspc)
        mean_fin_pos.append(fin_pos)
        mean_neg_size.append(fnc)
        mean_sec_neg_size.append(fsnc)
        mean_fin_neg.append(fin_neg)
        pos_nodes.append(pos_n_rep)
        print(cs, fpc, fspc, fin_pos, fnc, fsnc, fin_neg)

    save_experiment(nx_graph_creator.__name__, kwargs, p_linspace, mean_steps, mean_pos_size, mean_sec_pos_size, mean_fin_pos, 
                    mean_neg_size, mean_sec_neg_size, mean_fin_neg, pos_nodes)