from nas_bench_graph.architecture import Arch
from utils import key2structure

arch = Arch([0, 1, 2, 1], ['gcn', 'gin', 'fc', 'cheb'])

# test 
for _ in range(5000):
    arch.random_arch()
    if set(arch.ops) == {'skip'}:
        continue
    key_idx = arch.valid_hash()
    structure = key2structure(key_idx)
    arch_tmp = Arch(structure[0], structure[1])
    assert arch_tmp.valid_hash() == key_idx