from lefdef import C_LefReader, C_DefReader

def parse_design(def_file_path):
    """Load design data from DEF file"""
    def_reader = C_DefReader()
    def_data = def_reader.read("ispd18_sample/ispd18_sample.input.def")
    
    design = {
        'components': {},
        'nets': {}
    }
    
    components = def_data.c_components
    nets = def_data.c_nets

    # Parse components
    for i in range(def_data.c_num_components):
        comp_id = components[i].c_id
        if isinstance(comp_id, bytes):
            comp_id = comp_id.decode('utf-8')
        design['components'][comp_id] = {
            'id': comp_id,
            'x': components[i].c_x,
            'y': components[i].c_y,
            'nets': []
        }

    # Parse nets
    for i in range(def_data.c_num_nets):
        net_name = nets[i].c_name
        if isinstance(net_name, bytes):
            net_name = net_name.decode('utf-8')
        design['nets'][net_name] = {'components': []}
        
        for j in range(nets[i].c_num_pins):
            instance = nets[i].c_instances[j]
            if isinstance(instance, bytes):
                instance = instance.decode('utf-8')
            if instance in design['components']:
                design['components'][instance]['nets'].append(net_name)
                design['nets'][net_name]['components'].append(instance)
                
        # Remove duplicates
        design['nets'][net_name]['components'] = list(set(design['nets'][net_name]['components']))
    
    return design

