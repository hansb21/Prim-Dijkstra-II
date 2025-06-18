import parse 
import mst 
import pd 

def main():
    def_file_path = "ispd18_sample/ispd18_sample.input.def"
    design = parse.parse_design(def_file_path)
    
    G_mst = pd.prim(G_mst, source=1) 
    G_mst = mst.build_mst_per_net(design)
    print(f"Design contains {len(design['components'])} components")
    print(f"Design contains {len(design['nets'])} nets")
    print(f"MST contains {len(G_mst.edges())} connections")
    
    mst.draw_manhattan_mst(G_mst)

if __name__ == "__main__":
    main()

