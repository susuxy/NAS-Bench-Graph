import pickle
from nas_bench_graph.architecture import all_archs

def light_read(dname):
    f = open("nas_bench_graph/light/{}.bench".format(dname), "rb")
    bench = pickle.load(f)
    f.close()
    return bench

def read(name):
    f = open(name, "rb")
    bench = pickle.load(f)
    f.close()
    return bench

if __name__ == "__main__":
    read()
