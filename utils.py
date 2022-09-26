from nas_bench_graph.architecture import gnn_list

def link2lidx(lk):
    if lk == [0,0,0,0]:
        lk_hash = 0
    elif lk == [0,0,0,1]:
        lk_hash = 1
    elif lk == [0,0,1,1]:
        lk_hash = 2
    elif lk == [0,0,1,2]:
        lk_hash = 3
    elif lk == [0,0,1,3]:
        lk_hash = 4
    elif lk == [0,1,1,1]:
        lk_hash = 5
    elif lk == [0,1,1,2]:
        lk_hash = 6
    elif lk == [0,1,2,2]:
        lk_hash = 7
    elif lk == [0,1,2,3]:
        lk_hash = 8
    else:
        raise ValueError('link structure not exist')
    return lk_hash

def lidx2link(lk_hash):
    if lk_hash == 0:
        lk = [0,0,0,0]
    elif lk_hash == 1:
        lk = [0,0,0,1]
    elif lk_hash == 2:
        lk = [0,0,1,1]
    elif lk_hash == 3:
        lk = [0,0,1,2]
    elif lk_hash == 4:
        lk = [0,0,1,3]
    elif lk_hash == 5:
        lk = [0,1,1,1]
    elif lk_hash == 6:
        lk = [0,1,1,2]
    elif lk_hash == 7:
        lk = [0,1,2,2]
    elif lk_hash == 8:
        lk = [0,1,2,3]
    else:
        raise ValueError('link index not exist')
    return lk

gnn2gidx = {name: i for i, name in enumerate(gnn_list)}
gidx2gnn = {i: name for i, name in enumerate(gnn_list)}

# bench key to original structure

def key2structure(bench_key):
    link_structure = lidx2link(bench_key // 10000)
    ops0 = gidx2gnn[bench_key // 1000 - bench_key // 10000 * 10]
    ops1 = gidx2gnn[bench_key // 100 - bench_key // 1000 * 10]
    ops2 = gidx2gnn[bench_key // 10 - bench_key // 100 * 10]
    ops3 = gidx2gnn[bench_key // 1 - bench_key // 10 * 10]
    return [link_structure, [ops0, ops1, ops2, ops3]]


def dataset2info(dataset_name):
    if dataset_name.startswith('arxiv'):
        return [169343, 1166243, 128, 40]
    elif dataset_name.startswith('citeseer'):
        return [3327, 4732, 3703, 6]
    elif dataset_name.startswith('computers'):
        return [13381, 245778, 767, 10]
    elif dataset_name.startswith('cora'):
        return [2708, 5429, 1433, 7]
    elif dataset_name.startswith('cs'):
        return [18333, 81894, 6805, 15]
    elif dataset_name.startswith('photo'):
        return [7487, 119043, 745, 8]
    elif dataset_name.startswith('physics'):
        return [34493, 247962, 8415, 5]
    elif dataset_name.startswith('proteins'):
        return [132534, 39561252, 8, 112]
    elif dataset_name.startswith('pubmed'):
        return [19717, 44338, 500, 3]
