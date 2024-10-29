import igraph as ig
from msccl.language.ir import *
from msccl.language.rank_dag import *
from msccl.language.collectives import *
from collections import deque
from dataclasses import dataclass, field

import random
random.seed(1234)

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




def visualize_instruction_dag1(instruction_dag: InstructionDAG, collective: Collective, ):
        

        


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
    
    def add_chunk_node(buffer, rank, index, nnodes) -> int:
        nnodes += 1
        vertex_labels.append(format_label(buffer, rank, index))
        vertex_colors.append("yellow")
        vertex_shapes.append("rectangle")
        vertex_types.append("c")
        return nnodes
    
    def remove_chunk_node(num) -> int:
        nnodes -= 1
        del vertex_labels[num]
        del vertex_colors[num]
        del vertex_shapes[num]
        del vertex_types[num]
        return nnodes


    def add_chunk_nodes(buffer: str, rank: int, index: int, size: int, op_num: int, nnodes: int, read = True):
        added = []
        for i in range(index, index + size):
            nnodes = add_chunk_node(buffer, rank, i, nnodes)
            added.append(nnodes)
            if read: edges.append([nnodes, op_num])
            else: edges.append([op_num, nnodes])
        return nnodes, added
    
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
            last_writers[(op.src.buffer.value, op.src.rank, op.src.index)] = op.num 
            root_owner[str(op.num)] = (op.src.buffer.value, op.src.rank, op.src.index) # it will be inherited from all the consequent childs

            # if it is a starting operation, replace the operation node with the addressed chunk
            vertex_labels[op.num] = format_label(op.src.buffer, op.src.rank, op.src.index)
            vertex_colors[op.num] = "light blue"
            vertex_shapes[op.num] = "rectangle"
            vertex_types[op.num] = "c"
            

        elif op.inst == Instruction.send:
            last_writer = last_writers[(op.src.buffer.value, op.src.rank, op.src.index)] 

            # Inherit the root_owner form the last_writer operation
            root_owner[str(op.num)] = root_owner[str(last_writer)]

            nnodes, added = add_chunk_nodes(op.src.buffer, op.src.rank, op.src.index, op.src.size, op.num, nnodes, read=True)
            if root_owner in added:
                remove_chunk_node(root_owner)

            # add an edge from the last_writer operation of the src chunk to this node
            edges.append((last_writer, op.num))
            

        elif op.inst == Instruction.recv:
            edges.append([op.send_match.num, op.num]) # Add an edge from the sender
            nnodes, added = add_chunk_nodes(op.send_match.dst.buffer, op.send_match.dst.rank, op.send_match.dst.index, op.send_match.dst.size, op.num, nnodes, read=False)
            
            # Inherit the root_owner form the sender operation
            root_owner[str(op.num)] = root_owner[str(op.send_match.num)]


        elif op.inst == Instruction.copy:
            # Inherit the root_owner form the last writer operation
            root_owner[str(op.num)] = root_owner[str(last_writers[(op.src.buffer.value, op.src.rank, op.src.index)])]

            # add an edge from the last_writer operation of the src chunk to this node
            edges.append((last_writers[(op.src.buffer.value, op.src.rank, op.src.index)], op.num))
            
            # Update the writer of the destination chunk with the node of this operation
            last_writers[(op.dst.buffer.value, op.dst.rank, op.dst.index)] = op.num

            nnodes, added = add_chunk_nodes(op.dst.buffer, op.dst.rank, op.dst.index, op.dst.size, op.num, nnodes, read=False)


        elif op.inst == Instruction.reduce:
            root_owner[str(op.num)] = root_owner[str(last_writers[(op.src.buffer.value, op.src.rank, op.src.index)])]

            edges.append((last_writers[(op.src.buffer.value, op.src.rank, op.src.index)], op.num))

            nnodes, added = add_chunk_nodes(op.dst.buffer, op.dst.rank, op.dst.index, op.dst.size, op.num, nnodes, read=False)



        elif op.inst == Instruction.recv_copy_send:
            edges.append([op.send_match.num, op.num]) # Add an edge from the sender 

            # add the writed notes and an edge to it
            nnodes = add_chunk_node(op.send_match.dst.buffer, op.send_match.dst.rank, op.send_match.dst.index, nnodes)
            edges.append([op.num, nnodes])

            # Inherit the root_owner form the sender operation
            root_owner[str(op.num)] = root_owner[str(op.send_match.num)]


        elif op.inst == Instruction.recv_reduce_copy_send:
            # Inherit the root_owner form the sender operation
            root_owner[str(op.num)] = root_owner[str(op.send_match.num)]

            # buffer to reduce the previous node with
            sender_buffer = op.send_match.dst.buffer
            sender_rank = op.send_match.dst.rank
            sender_index = op.send_match.dst.index
            size = op.send_match.dst.size # same for writing and recived

            # Writing buffer
            writing_buffer = op.recv_match.src.buffer
            writing_rank = op.recv_match.src.rank
            writing_index = op.recv_match.src.index

            edges.append([op.send_match.num, op.num]) # Add an edge from the sender 
            nnodes, added = add_chunk_nodes(sender_buffer, sender_rank, sender_index, size, op.num, nnodes, read=True) # the chunk reducing with

            # If the nodes to reduce with are the same as the write nodes:
            if sender_buffer == writing_buffer and sender_rank == writing_rank and sender_index == writing_index:
                for i in added:
                    edges.append([op.num, i])
            else:
                nnodes, added = add_chunk_nodes(writing_buffer, writing_rank, writing_index, size, op.num, nnodes, read = False) 
            if root_owner in added:
                remove_chunk_node(root_owner)



        elif op.inst == Instruction.recv_reduce_send:
            # Inherit the root_owner form the sender operation
            root_owner[str(op.num)] = root_owner[str(op.send_match.num)]

            sender_buffer = op.send_match.dst.buffer
            sender_rank = op.send_match.dst.rank
            sender_index = op.send_match.dst.index
            size = op.send_match.dst.size

            edges.append([op.send_match.num, op.num]) # Add an edge from the sender 

            nnodes, added = add_chunk_nodes(sender_buffer, sender_rank, sender_index, size, op.num, nnodes, read=True)
            if root_owner in added:
                remove_chunk_node(root_owner)




        return nnodes

    operations = instruction_dag.operations

    # slot -> op.num
    last_writers = {(buffer, rank, index):None for rank in range(collective.num_ranks) for index in range(collective.num_ranks*collective.chunk_factor) for buffer in ["i", "o", "s"]}


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
    root_owner = {str(node):None for node in range(nnodes)} # node -> root chunk that determines which chunk dag it belongs to
    
    for slot, ops in operations.items():
        frontier = [ops]
        while len(frontier) > 0:
            op = frontier[0]
            if all(dep in visited for dep in op.depends) and ((op.send_match in visited) or op.send_match == None):
                if op not in visited:
                    nnodes = add_node(op, nnodes)
                    visited.add(op)
                for next_op in op.next:
                    if all(dep in visited for dep in op.depends) and ((next_op.send_match in visited) or next_op.send_match == None):
                        if next_op not in visited:
                            nnodes = add_node(next_op, nnodes)
                            visited.add(next_op)
                frontier = frontier[1:] + op.next

            else:
                frontier = frontier[1:] + [op]

    g = ig.Graph(nnodes, edges, directed=True)
    g.vs["type"] = vertex_types
    g.vs["labels"] = vertex_labels
    g.vs["colors"] = vertex_colors
    g.vs["shapes"] = vertex_shapes
    g.vs["types"] = vertex_types
    g.delete_vertices([v.index for v in g.vs if g.degree(v.index) == 0])
    # layout = build_layout(g, nnodes, roots)

    

    # g = ig.Graph(nnodes, edges, directed=True)
    from math import log10
    style = {
    "vertex_label_size": 4,  
    "vertex_label_dist": 0,  
    "vertex_size": 10,  # Increase size
    "vertex_width": 17,
    "edge_width": 1,  
    "layout": "auto",
    "vertex_color": g.vs["colors"],
    "vertex_label": g.vs["labels"],
    "vertex_shape": g.vs["shapes"],
    "bbox": (nnodes * 10, nnodes * 10)
    }
    ig.plot(g, **style, target="ciao.pdf")






def visualize_instruction_dag_multiple(instruction_dag: InstructionDAG, collective: Collective, ):
    def format_label(buffer, rank, index):
        'Creates a representative string for the chunk'
        superscript = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
        subscript = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
        return f"C_{buffer}{str(rank).translate(superscript)}{str(index).translate(subscript)}"

    def assign_op_nodes(operations: list):
        # Puts all the operations except the starting ones in a list without keeping the gerarchy of the rank execution,
        # which is not necessary since we will analyze the various dependencies.
        # Assigns a number to each node and returns the number of assigned nodes
        # Returns the id of the root nodes and adds them to a visited list which is also returned
        nnodes = -1
        visited = set()
        operations_result = []
        vertex_labels = []
        vertex_colors = []
        vertex_shapes = []
        vertex_types = []
        visited_result = set()
        roots = {}
        last_writers = {}

        for slot, ops in operations.items():
            frontier = [ops]
            while len(frontier) > 0:
                op = frontier[0]
                if op not in visited:
                    visited.add(op)

                    # Preemptively add starting nodes to visited sets and set them as root
                    if op.inst == Instruction.start:
                        nnodes = add_start_node(op, nnodes, vertex_shapes, vertex_colors, vertex_labels, vertex_types)
                        roots.update({(op.src.buffer, op.src.rank, op.src.index) : op})
                        visited_result.add(op)
                        last_writers.update({(op.src.buffer, op.src.rank, op.src.index) : op})
                    else:
                        operations_result.append(op)
                        nnodes = add_op_node(op, nnodes, vertex_shapes, vertex_colors, vertex_labels, vertex_types)
                for next_op in op.next:
                    if next_op not in visited:
                        operations_result.append(next_op)
                        visited.add(next_op)
                        nnodes = add_op_node(next_op, nnodes, vertex_shapes, vertex_colors, vertex_labels, vertex_types)
                frontier = frontier[1:] + op.next

        return nnodes, operations_result, visited_result, roots, last_writers, vertex_shapes, vertex_colors, vertex_labels, vertex_types
    
    def add_chunk_node(buffer, rank, index, nnodes, vertex_shapes, vertex_colors, vertex_labels, vertex_types) -> int:
        nnodes += 1
        vertex_labels.append(format_label(buffer, rank, index))
        vertex_colors.append("yellow")
        vertex_shapes.append("rectangle")
        vertex_types.append("c")
        return nnodes
    
    def remove_node(num, vertex_shapes, vertex_colors, vertex_labels, vertex_types) -> int:
        nnodes -= 1
        del vertex_labels[num]
        del vertex_colors[num]
        del vertex_shapes[num]
        del vertex_types[num]
        return nnodes


    def add_chunk_nodes(buffer: str, rank: int, index: int, size: int, op_num: int, nnodes: int, vertex_shapes, vertex_colors, vertex_labels, vertex_types, read = True, root=None):
        '''
        adds some chunk nodes

        '''
        added = []
        for i in range(index, index + size):
            chunk = (buffer, rank, i)
            # don't add the chunk node relative to the root of the tree if it is a read operation
            if read and root != None and (chunk in roots and roots[chunk].num == root): continue
            nnodes = add_chunk_node(buffer, rank, i, nnodes, vertex_shapes, vertex_colors, vertex_labels, vertex_types)
            added.append(nnodes)
            if read: edges.append([nnodes, op_num])
            else: edges.append([op_num, nnodes])
        return nnodes, added
    
    def add_start_node(op, nnodes, vertex_shapes, vertex_colors, vertex_labels, vertex_types):
        nnodes = add_chunk_node(op.src.buffer, op.src.rank, op.src.index, nnodes, vertex_shapes, vertex_colors, vertex_labels, vertex_types)
        vertex_types[-1] = "s"
        op.num = [nnodes]
        op.roots = [(op.src.buffer, op.src.rank, op.src.index)]
        return nnodes
    
    def add_op_node(op: Op, nnodes, vertex_shapes, vertex_colors, vertex_labels, vertex_types):
        op.num = list(range(nnodes+1, nnodes+1+op.src.size))
        nnodes += op.src.size
        vertex_labels += [op.inst.value]*op.src.size
        vertex_colors += [instruction_color_mapping[op.inst]]*op.src.size
        vertex_shapes += ["circle"]*op.src.size
        vertex_types += ["op"]*op.src.size
        return nnodes
    
    def processed_dep(op) -> bool:
        if (op.send_match in visited or op.send_match == None):
            if all(dep in visited for dep in op.depends):
                return True
        return False
    
    def root(op):
        if op.inst == Instruction.start:
            root = op.roots[0]
        else: return "well"
    
    def connect_generate_chunk(op: Op, nnodes: int, vertex_shapes, vertex_colors, vertex_labels, vertex_types):
        last_writers

        if op.is_send():
            # Connect it to the last writer of the nodes

            # If it is sending multiple chunks and some of them are roots,
            # generate a operation node for each of the roots and attach it to
            # the last writer of the relative root graph

            # Iterate trough the different nodes representing this op
            for i, num in enumerate(op.num):
                # iterate trough the src chunk of the operation
                for index in range(op.src.index, op.src.index + op.src.size):
                    chunk = (op.src.buffer, op.src.rank, index)
                    op.roots.append(root(last_writers[chunk]))
                    if chunk in roots: continue
                    nnodes = add_chunk_node(*chunk, nnodes, vertex_shapes, vertex_colors, vertex_labels, vertex_types)
                    edges.append(nnodes, num)

                        
                    # nnodes, added = add_chunk_nodes(chunk[0], chunk[1], chunk[2], op.src.size, op.num, nnodes, vertex_shapes, vertex_colors, vertex_labels, vertex_types, read=True, root=root)
                
        return nnodes

                    

            
    
    


    # chunks_in = {(buffer, rank, index):[] for rank in range(collective.num_ranks) for index in range(collective.num_ranks*collective.chunk_factor) for buffer in ["i", "o"]}
    # chunks_out = {(buffer, rank, index):[] for rank in range(collective.num_ranks) for index in range(collective.num_ranks*collective.chunk_factor) for buffer in ["i", "o"]}
    # last_readers = {(rank, buffer, index):[] for rank in range(collective.num_ranks) for index in range(collective.num_ranks*collective.chunk_factor) for buffer in ["i", "o"]}


    edges = []

    nnodes, operations, visited, roots, last_writers, vertex_shapes, vertex_colors, vertex_labels, vertex_types = assign_op_nodes(instruction_dag.operations)
    # nnodes -> total number of nodes in graph [initialized to the number of operations]
    # operations -> list with all the operations still to visit [initialized with the non 'st' operations]
    # visited -> set with the currently visited operations [initialized with the 'st' operations]
    # roots -> (slot -> op) for each root chunk the 'st' operation 
    # last_writers -> (slot -> op) for each chunk the operation that wrote it last

    for op in operations:
        if processed_dep(op):
            visited.add(op)
            nnodes = connect_generate_chunk(op, nnodes, vertex_shapes, vertex_colors, vertex_labels, vertex_types)     


    g = ig.Graph(nnodes, edges, directed=True)
    g.vs["type"] = vertex_types
    g.vs["labels"] = vertex_labels
    g.vs["colors"] = vertex_colors
    g.vs["shapes"] = vertex_shapes
    g.vs["types"] = vertex_types
    # g.delete_vertices([v.index for v in g.vs if g.degree(v.index) == 0])
    # layout = build_layout(g, nnodes, roots)

    

    # g = ig.Graph(nnodes, edges, directed=True)
    from math import log10
    style = {
    "vertex_label_size": 4,  
    "vertex_label_dist": 0,  
    "vertex_size": 10,  # Increase size
    "vertex_width": 17,
    "edge_width": 1,  
    "layout": "auto",
    "vertex_color": g.vs["colors"],
    "vertex_label": g.vs["labels"],
    "vertex_shape": g.vs["shapes"],
    "bbox": (nnodes * 10, nnodes * 10)
    }
    ig.plot(g, **style, target="ciao.pdf")


def visualize_instruction_dag(instruction_dag: InstructionDAG, collective: Collective, ):
    edges = []
    vertex_labels = []
    vertex_colors = []
    vertex_shapes = []
    vertex_types = []
    vertex_height = []
    # roots -> (slot -> op) for each root chunk the 'st' operation 
    roots = {}
    # last_writers -> (slot -> op) for each chunk the operation that wrote it last
    last_writers = {}

    def format_label(buffer, rank, index):
        'Creates a representative string for the chunk'
        superscript = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
        subscript = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
        return f"C_{buffer}{str(rank).translate(superscript)}{str(index).translate(subscript)}"
    
    def remove_zero_degree_nodes(g):
        zero_degree_indices = [i for i, d in enumerate(g.degree()) if d == 0]
    
        # Remove the zero-degree nodes
        g.delete_vertices(zero_degree_indices)
        
        return g
    
    
    def add_chunk_node(buffer, rank, index, height, nnodes) -> int:
        nnodes += 1
        vertex_labels.append(format_label(buffer, rank, index))
        vertex_colors.append("yellow")
        vertex_shapes.append("rectangle")
        vertex_types.append("c")
        vertex_height.append(height)
        return nnodes
            
    def add_op_node(op, nnodes, height = -1):
        nnodes += 1
        op.num = nnodes
        vertex_labels.append(op.inst.value)
        vertex_colors.append(instruction_color_mapping[op.inst])
        vertex_shapes.append("circle")
        vertex_types.append("op")
        vertex_height.append(height)
        return nnodes
        
    def add_start_node(op: Op, nnodes):
        nnodes = add_chunk_node(op.src.buffer, op.src.rank, op.src.index, 0, nnodes)
        vertex_types[-1] = "s"
        vertex_colors[-1] = "violet"
        op.num = nnodes
        op.root = (op.src.buffer, op.src.rank, op.src.index)
        return nnodes

    def processed_dep(op) -> bool:
        # Maybe: fix when a dependence is that the src chunk is never writter
        if (op.send_match in visited or op.send_match == None):
            if all(dep in visited for dep in op.depends):
                return True
        return False
    
    def populate_non_root_chunks(collective, nnodes):
        non_root_chunks = {}
        
        for buffer in [Buffer.output]:
            for rank in range(collective.num_ranks):
                for index in range(collective.chunk_factor):
                    key = (buffer, rank, index)
                    non_root_chunks[key] = add_chunk_node(buffer, rank, index, 0, nnodes)
                    nnodes += 1
        
        return non_root_chunks, nnodes
    
    def assign_op_nodes(operations: list):
        # Puts all the operations except the starting ones in a list without keeping the gerarchy of the rank execution,
        # which is not necessary since the informations of the various operations are stored in each operation
        # Assigns a number to each node and returns the number of assigned nodes
        # Returns the id of the root nodes and adds them to a visited list which is also returned
        # And creates the nodes for the starting nodes
        nnodes = -1
        visited = set()
        operations_result = []
        visited_result = set()

        for slot, ops in operations.items():
            frontier = [ops]
            while len(frontier) > 0:
                op: Op = frontier[0]
                if op not in visited:
                    visited.add(op)

                    # Preemptively add starting nodes to visited sets and set them as root
                    if op.inst == Instruction.start:
                        nnodes = add_start_node(op, nnodes)
                        roots.update({(op.src.buffer, op.src.rank, op.src.index) : op})
                        visited_result.add(op)
                        last_writers.update({(op.src.buffer, op.src.rank, op.src.index) : op})
                    else:
                        operations_result.append(op)
                        nnodes = add_op_node(op, nnodes)
                for next_op in op.next:
                    if next_op not in visited:
                        operations_result.append(next_op)
                        visited.add(next_op)
                        nnodes = add_op_node(next_op, nnodes)
                frontier = frontier[1:] + op.next

        return nnodes, operations_result, visited_result    

    
    def connect_generate_chunk(op: Op, nnodes):
        vertex_labels
        if op.inst == Instruction.recv_copy_send:
            vertex_labels
        if op.is_local():
            # Connect to previous op and genereate chunk nodes
            src_chunk = (op.src.buffer, op.src.rank, op.src.index)
            last_writer = last_writers[src_chunk]
            # Check if the chunks this op is sending are a superset of the chunks the last writer wrote
            # If it is:
            # Connect this op to the last writer op
            # If it is not:
            # Connect this op to one of the source chunks (say A)
            # Anyway, generate chunk nodes that are not in A and connect them to this op node
            if last_writer.dst.size > 1:
                chunk_index_last_writer = set(range(last_writer.dst.index, last_writer.dst.index + last_writer.dst.size))
            else:
                chunk_index_last_writer = {last_writer.dst.index}
            if op.src.size > 1:
                chunk_index_src = set(range(op.src.index, op.src.index + op.src.size))
            else:
                chunk_index_src = {op.src.index}
            if chunk_index_src.issuperset(chunk_index_last_writer): 
                edges.append([last_writer.num, op.num])
                wrote_chunks_by_previous_op_node = chunk_index_last_writer
            else:
                # Fix in the case the src chunk is not an input chunk
                if src_chunk in roots:
                    edges.append([roots[src_chunk].num, op.num])
                else:
                    edges.append([non_root_chunks[src_chunk], op.num])
                wrote_chunks_by_previous_op_node = {op.src.index}
            vertex_height[op.num] = vertex_height[last_writer.num] + 1
            for chunk_index in chunk_index_src.difference(wrote_chunks_by_previous_op_node):
                nnodes = add_chunk_node(op.src.buffer, op.src.rank, chunk_index, vertex_height[op.num], nnodes)
                edges.append([nnodes, op.num])

        if op.is_recv():
            # Connect this node to the sender and set the height
            edges.append([op.send_match.num, op.num])
            vertex_height[op.num] = vertex_height[op.send_match.num] + 1
            
        if op.is_write():
            # Add chunk nodes being written

            if op.is_send():
                # Writing the dst chunk of the send_match   
                # Write and send operations are rcs, rrs, rrcs
                # They write on the dst chunk of the send_match
                for chunk_index in range(op.send_match.dst.index, op.send_match.dst.index + op.send_match.dst.size):
                    last_writers[(op.send_match.dst.buffer, op.send_match.dst.rank, op.send_match.dst.index)] = op
                    nnodes = add_chunk_node(op.send_match.dst.buffer, op.send_match.dst.rank, chunk_index, vertex_height[op.num], nnodes)
                    edges.append([op.num, nnodes])
            else:
                # Writing in the dst of this chunk
                for chunk_index in range(op.dst.index, op.dst.index + op.dst.size):
                    last_writers[(op.dst.buffer, op.dst.rank, op.dst.index)] = op
                    nnodes = add_chunk_node(op.dst.buffer, op.dst.rank, chunk_index, vertex_height[op.num], nnodes)
                    edges.append([op.num, nnodes])

        if op.is_send() and not op.is_recv():
            src_chunk = (op.src.buffer, op.src.rank, op.src.index)
            last_writer = last_writers[src_chunk]
            # Check if the chunks this op is sending are a superset of the chunks the last writer wrote
            # If it is:
            # Connect this op to the last writer op
            # If it is not:
            # Connect this op to one of the source chunks (say A)
            # Anyway, generate chunk nodes that are not in A and connect them to this op node
            if last_writer.dst.size > 1:
                chunk_index_last_writer = set(range(last_writer.dst.index, last_writer.dst.index + last_writer.dst.size))
            else: # Without this it would say {index + 1}
                chunk_index_last_writer = {last_writer.dst.index}
            if op.src.size > 1:
                chunk_index_src = set(range(op.src.index, op.src.index + op.src.size))
            else:
                chunk_index_src = {op.src.index}
            if chunk_index_src.issuperset(chunk_index_last_writer): 
                edges.append([last_writer.num, op.num])
                wrote_chunks_by_previous_op_node = chunk_index_last_writer
            else:
                # Fix in the case the src chunk is not an input chunk
                if src_chunk in roots:
                    edges.append([roots[src_chunk].num, op.num])
                else:
                    edges.append([non_root_chunks[src_chunk], op.num])
                wrote_chunks_by_previous_op_node = {op.src.index}
            vertex_height[op.num] = vertex_height[last_writer.num] + 1
            for chunk_index in chunk_index_src.difference(wrote_chunks_by_previous_op_node):
                nnodes = add_chunk_node(op.src.buffer, op.src.rank, chunk_index, vertex_height[op.num], nnodes)
                edges.append([nnodes, op.num])

        if op.inst == Instruction.recv_reduce_send:
            for chunk_index in range(op.send_match.dst.index, op.send_match.dst.index + op.send_match.dst.size):
                    last_writers[(op.send_match.dst.buffer, op.send_match.dst.rank, op.send_match.dst.index)] = op
                    nnodes = add_chunk_node(op.send_match.dst.buffer, op.send_match.dst.rank, chunk_index, vertex_height[op.num], nnodes)
                    edges.append([nnodes, op.num])

        return nnodes
    

    nnodes, operations, visited = assign_op_nodes(instruction_dag.operations)
    # nnodes -> total number of nodes in graph [initialized to the number of operations]
    # operations -> list with all the operations still to visit [initialized with the non 'st' operations]
    # visited -> set with the currently visited operations [initialized with the 'st' operations]


    # slot -> op representing
    non_root_chunks, nnodes = populate_non_root_chunks(collective, nnodes)


    while len(operations) > 0:
        op = operations[0]

        if processed_dep(op):
            visited.add(op)
            nnodes = connect_generate_chunk(op, nnodes)    
            operations = operations[1:]

        else: operations = operations[1:] + [op]


    g = ig.Graph(nnodes, edges, directed=True)
    g.vs["type"] = vertex_types
    g.vs["labels"] = vertex_labels
    g.vs["colors"] = vertex_colors
    g.vs["shapes"] = vertex_shapes
    g.vs["types"] = vertex_types
    
    g = remove_zero_degree_nodes(g)
    
    style = {
    "vertex_label_size": 4,  
    "vertex_label_dist": 0,  
    "vertex_size": 10,  # Increase size
    "vertex_width": 17,
    "edge_width": 1,  
    "layout": "auto",
    "vertex_color": g.vs["colors"],
    "vertex_label": g.vs["labels"],
    "vertex_shape": g.vs["shapes"],
    "bbox": (nnodes * 10, nnodes * 10)
    }
    ig.plot(g, **style, target="ciao.pdf")
         
