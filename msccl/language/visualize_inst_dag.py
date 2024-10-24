import igraph as ig
from msccl.language.ir import *
from msccl.language.rank_dag import *
from msccl.language.collectives import *
from collections import deque
from dataclasses import dataclass, field

def infer_init_buffers(collective_name):
    if collective_name == "allgather":
        pass
    elif collective_name == "allreduce":
        pass
    elif collective_name == "reduce_scatter":
        pass
    elif collective_name == "alltoall":
        pass

instruction_color_mapping = {
    Instruction.start: "white",
    Instruction.send: "white",
    Instruction.recv: "white",
    Instruction.copy: "white",
    Instruction.reduce: "white",
    Instruction.nop: "white",
    Instruction.delete: "white",
    
    # Fused instructions mapped to light green
    Instruction.recv_reduce_copy: "light green",
    Instruction.recv_reduce_copy_send: "light green",
    Instruction.recv_reduce_send: "light green",
    Instruction.recv_copy_send: "light green",
    }

def build_layout(g, nnodes, roots):
    def assign_positions(graph, root, start_x, start_y, x_spacing=1, y_spacing=1):
        """
        Assigns positions to the nodes in the component rooted at 'root'.
        'start_x' and 'start_y' define the starting position for this component.
        Returns a dictionary of positions.
        """
        from collections import deque
        
        positions = {}
        queue = deque()
        queue.append((root, start_x, start_y))
        positions[root] = (start_x, start_y)
        
        # To keep track of next available x position for 'c' nodes
        current_x = start_x
        # To keep track of y levels
        level_y = start_y + y_spacing
        
        while queue:
            current_node, x, y = queue.popleft()
            neighbors = graph.neighbors(current_node)
            for neighbor in neighbors:
                if neighbor not in positions:
                    neighbor_type = graph.vs[neighbor]["type"]
                    if neighbor_type == 'op':
                        # Place 'op' nodes vertically below
                        positions[neighbor] = (x, y + y_spacing)
                    elif neighbor_type == 'c':
                        # Place 'c' nodes to the left
                        positions[neighbor] = (x - x_spacing, y)
                    queue.append((neighbor, positions[neighbor][0], positions[neighbor][1]))
        
        return positions

    # Initialize overall positions
    overall_positions = {}

    # Define spacing
    x_offset = 0
    y_start = 0
    x_spacing = 10
    y_spacing = 5

    # Process each root separately
    for root in roots:
        positions = assign_positions(g, root, x_offset, y_start, x_spacing, y_spacing)
        # Update overall positions
        overall_positions.update(positions)
        # Update x_offset for next component
        # Estimate the width of the current component
        xs = [pos[0] for pos in positions.values()]
        width = max(xs) - min(xs) + x_spacing
        x_offset += width

    # Extract layout
    layout = []
    for v in range(nnodes+1):
        if v in overall_positions:
            layout.append(overall_positions[v])
        else:
            # For disconnected nodes, place them separately
            layout.append((x_offset, y_start))
            x_offset += x_spacing

    # Normalize layout coordinates for better visualization
    x_coords, y_coords = zip(*layout)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    canvas_scale_factor = nnodes / 50  
    normalized_layout = [((x - min_x) / max((max_x - min_x), 1) * canvas_scale_factor, 
                        (y - min_y) / max((max_y - min_y), 1) * canvas_scale_factor) for x, y in layout]
    return normalized_layout

def infer_instruction_dag_from_rank_dag():
    pass




def visualize_instruction_dag(instruction_dag: InstructionDAG, collective: Collective, ):

    def format_label(buffer, rank, index):
        'Creates a representative string for the chunk'
        superscript = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
        subscript = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
        return f"C_{buffer}{str(rank).translate(superscript)}{str(index).translate(subscript)}"

    def assign_op_nodes():
        nnodes = -1
        visited = set()
        for slot, ops in operations.items():
            frontier = [ops]
            while len(frontier) > 0:
                op = frontier[0]
                if op not in visited:
                    visited.add(op)
                    nnodes = add_op_node(op, nnodes)
                for next_op in op.next:
                    if next_op not in visited:
                        visited.add(next_op)
                        nnodes = add_op_node(next_op, nnodes)
                frontier = frontier[1:] + op.next
        return nnodes
    
    def add_chunk_node(buffer, rank, index, nnodes):
        nnodes += 1
        vertex_labels.append(format_label(buffer, rank, index))
        vertex_colors.append("yellow")
        vertex_shapes.append("rectangle")
        vertex_types.append("c")
        return nnodes
    
    def add_op_node(op, nnodes):
        nnodes += 1
        op.num = nnodes
        vertex_labels.append(op.inst.value)
        vertex_colors.append(instruction_color_mapping[op.inst])
        vertex_shapes.append("circle")
        vertex_types.append("op")
        return nnodes
       
    def add_node(op: Op, nnodes):
        # The initial nodes of the instruction_dag are the starting operations
        if op.inst == Instruction.start:
            roots.update({(op.src.buffer.value, op.src.rank, op.src.index) : op.num })
            last_writer[(op.src.buffer.value, op.src.rank, op.src.index)] = op.num
            
            # if it is a starting operation, replace the operation node with the addressed chunk
            vertex_labels[op.num] = format_label(op.src.buffer, op.src.rank, op.src.index)
            vertex_colors[op.num] = "light blue"
            vertex_shapes[op.num] = "rectangle"
            vertex_types[op.num] = "c"
            

        elif op.inst == Instruction.send:
            # add an edge from the last_writer operation of the src chunk to this node
            edges.append((last_writer[(op.src.buffer.value, op.src.rank, op.src.index)], op.num))
            pass

        elif op.inst == Instruction.recv:
            edges.append([op.send_match.num, op.num]) # Add an edge from the sender
            nnodes = add_chunk_node(op.send_match.dst.buffer, op.send_match.dst.rank, op.send_match.dst.index, nnodes)
            edges.append([op.num, nnodes])

        elif op.inst == Instruction.copy:
            # add an edge from the last_writer operation of the src chunk to this node
            edges.append((last_writer[(op.src.buffer.value, op.src.rank, op.src.index)], op.num))
            
            # Update the writer of the destination chunk with the node of this operation
            last_writer[(op.dst.buffer.value, op.dst.rank, op.dst.index)] = op.num

            nnodes = add_chunk_node(op.dst.buffer, op.dst.rank, op.dst.index, nnodes)
            edges.append([op.num, nnodes])

        elif op.inst == Instruction.recv_copy_send:
            edges.append([op.send_match.num, op.num]) # Add an edge from the sender 
            nnodes = add_chunk_node(op.send_match.dst.buffer, op.send_match.dst.rank, op.send_match.dst.index, nnodes)
            edges.append([op.num, nnodes])


        return nnodes

    operations = instruction_dag.operations

    # slot -> op.num
    last_writer = {(buffer, rank, index):None for rank in range(collective.num_ranks) for index in range(collective.num_ranks*collective.chunk_factor) for buffer in ["i", "o"]}

    root_belong = {}

    chunks_in = {(buffer, rank, index):[] for rank in range(collective.num_ranks) for index in range(collective.num_ranks*collective.chunk_factor) for buffer in ["i", "o"]}
    chunks_out = {(buffer, rank, index):[] for rank in range(collective.num_ranks) for index in range(collective.num_ranks*collective.chunk_factor) for buffer in ["i", "o"]}
    # last_readers = {(rank, buffer, index):[] for rank in range(collective.num_ranks) for index in range(collective.num_ranks*collective.chunk_factor) for buffer in ["i", "o"]}
    visited = set()


    roots = {}
    edges = []
    vertex_labels = []
    vertex_colors = []
    vertex_shapes = []
    vertex_types = []


    nnodes = assign_op_nodes() # Number the operation nodes starting by counting from 0
    
    for slot, ops in operations.items():
        frontier = [ops]
        while len(frontier) > 0:
            op = frontier[0]
            if op not in visited:
                nnodes = add_node(op, nnodes)
                visited.add(op)
            for next_op in op.next:
                if next_op not in visited:
                    nnodes = add_node(next_op, nnodes)
                    visited.add(next_op)


            frontier = frontier[1:] + op.next

    g = ig.Graph(nnodes, edges, directed=False)
    g.vs["type"] = vertex_types
    layout = build_layout(g, nnodes, roots)

    

    g = ig.Graph(nnodes, edges, directed=True)
    from math import log10
    style = {
    "vertex_label_size": 4,  
    "vertex_label_dist": 0,  
    "vertex_size": 10,  # Increase size
    "vertex_width": 17,
    "edge_width": 1,  
    "layout": "auto",
    "vertex_color": vertex_colors,
    "vertex_label": vertex_labels,
    "vertex_shape": vertex_shapes,
    "bbox": (nnodes * 10, nnodes * 10)
    }
    ig.plot(g, **style, target="ciao.pdf")