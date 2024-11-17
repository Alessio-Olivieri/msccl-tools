import igraph as ig
from msccl.language.ir import *
from msccl.language.rank_dag import *
from msccl.language.collectives import *

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

def visualize_instruction_dag(instruction_dag: InstructionDAG, collective: Collective):
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
    def argmin(a):
        return min(range(len(a)), key=lambda x : a[x])
    def argmax(a):
        return max(range(len(a)), key=lambda x : a[x])

    def format_label(buffer, rank, index):
        'Creates a representative string for the chunk'
        superscript = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
        subscript = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
        return f"C_{buffer}{str(rank).translate(superscript)}{str(index).translate(subscript)}"
    
    def decide_last_writer(buffer, rank, index, size):
        # The last writer is the operation that wrote the chunk for last.
        # If the chunk size is bigger than one we consider the operation that
        # wrote a chunk the last.
        # If there are many of them consider numerical order: (rank,  index)

        chunks = [(buffer, rank, i) for i in range(index, index + size)]
        last_writers_buffer_op = [last_writers[chunk] for chunk in chunks]
        last_writers_buffer_height = [vertex_height[last_writers[chunk].num] for chunk in chunks]
        last_writer = last_writers_buffer_op[last_writers_buffer_height.index(max(last_writers_buffer_height))]
        last_chunk = chunks[last_writers_buffer_height.index(max(last_writers_buffer_height))]
        return last_chunk, last_writer

    
    def remove_zero_degree_nodes(g: ig.Graph):
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
                if (op.src.buffer, op.src.rank, op.src.index) in last_writers:
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

        op: Op
        # Populate scratch buffers:
        for op in operations:
            if op.dst.buffer == Buffer.scratch:
                for index in range(op.dst.index, op.dst.index + op.dst.size):
                    key = (Buffer.scratch, op.dst.rank, index)
                    if key not in non_root_chunks:
                        non_root_chunks.update({key : add_chunk_node(Buffer.scratch, op.dst.rank, index, 0, nnodes)})
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
            src_chunk, last_writer = decide_last_writer(op.src.buffer, op.src.rank, op.src.index, op.src.size)
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
                    chunk = (op.send_match.dst.buffer, op.send_match.dst.rank, chunk_index)
                    if chunk in last_writers:
                        last_writers[chunk] = op
                    else:
                        last_writers.update({chunk : op})
                    nnodes = add_chunk_node(op.send_match.dst.buffer, op.send_match.dst.rank, chunk_index, vertex_height[op.num], nnodes)
                    edges.append([op.num, nnodes])
            else:
                # Writing in the dst of this chunk
                for chunk_index in range(op.dst.index, op.dst.index + op.dst.size):
                    chunk = (op.dst.buffer, op.dst.rank, chunk_index)
                    if chunk in last_writers:
                        last_writers[chunk] = op
                    else:
                        last_writers.update({chunk : op})
                    nnodes = add_chunk_node(op.dst.buffer, op.dst.rank, chunk_index, vertex_height[op.num], nnodes)
                    edges.append([op.num, nnodes])

        if op.is_send() and not op.is_recv():
            # src_chunk = (op.src.buffer, op.src.rank, op.src.index)
            # last_writer = last_writers[src_chunk]
            src_chunk, last_writer = decide_last_writer(op.src.buffer, op.src.rank, op.src.index, op.src.size)
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
                    chunk = (op.send_match.dst.buffer, op.send_match.dst.rank, chunk_index)
                    if chunk in last_writers:
                        last_writers[chunk] = op
                    else:
                        last_writers.update({chunk : op})
                    nnodes = add_chunk_node(op.send_match.dst.buffer, op.send_match.dst.rank, chunk_index, vertex_height[op.num], nnodes)
                    edges.append([nnodes, op.num])

        return nnodes
    
    def build_layout(g: ig.Graph):
        def flatten(xss):
            return [x for xs in xss for x in xs]
        def infer_next_ops(node):
            return [neighbor for neighbor in g.neighbors(node, mode="out")
                        if g.vs[neighbor]["type"] == "op"]
        def infer_read_chunks(node):
            return [neighbor for neighbor in g.neighbors(node, mode="in")
                                    if g.vs[neighbor]["type"] == "c" and neighbor not in roots]
        def infer_write_chunks(node):
            return [neighbor for neighbor in g.neighbors(node, mode="out")
                                    if g.vs[neighbor]["type"] == "c" and neighbor not in roots]
        def infer_leafs_count(root):
            next_ops = infer_next_ops(root)
            if next_ops == []:
                return 1
            else:
                leafcount = 0
                for neighbor in next_ops:  # Get the neighbors of the current vertex
                    leafcount += infer_leafs_count(neighbor)
                leaf_count[root] = leafcount
                return leafcount
            
        def find_next_ops_x(root, child_count):
            selected = []
            offset = 0  # This will help alternate numbers on either side of the target

            if child_count%2 == 1:
                selected.append(root)
            
            # Loop until we find the required number of close numbers
            while len(selected) < child_count:
                # Alternate picking numbers above and below the target
                if offset % 2 == 0:
                    candidate = root + (offset // 2) * 2 +1
                else:
                    candidate = root - (offset // 2 + 1) * 2 +1
                
                # Check if candidate meets the min distance requirement with all selected numbers
                if all(abs(candidate - sel) >= 2 for sel in selected):
                    selected.append(candidate)
                
                offset += 1  # Move to the next offset
            return selected
            
        def set_x_op(root, start_x):
            x[root] = start_x + ((leaf_count[root]*3)//2)
            next_ops = infer_next_ops(root)
            for neighbor in next_ops:  # Get the neighbors of the current vertex
                set_x_op(neighbor, start_x)

        x = [-1] * len(g.vs)
        y = [-1] * len(g.vs)
        roots = [i for i, height in enumerate(g.vs["steps"]) if height==0]
        leaf_count = [-1]*len(g.vs)
        for root in roots:
            infer_leafs_count(root) # saves in leaf_count
            y[root] = 0

        # Set x for the root nodes
        prev_child_pos = 0 # the starting position of the childs on the left
        for root in roots:
            x[root] = prev_child_pos + (leaf_count[root] * 3 // 2)
            prev_child_pos += leaf_count[root] * 3

        # set the other nodes
        queue = roots.copy()
        floor = 0
        to_set_y = []
        current_height = 0
        while queue:
            new_queue = []
            added_chunks = [] # number of chunks interacting with each op node of this floor

            # set positions for chunks in and out
            for op_node in queue:
                chunks_in = infer_read_chunks(op_node)
                for i, chunk in enumerate(chunks_in):
                    x[chunk] = x[op_node] - 1
                    y[chunk] = y[op_node] + i / 5
                added_chunks.append(len(chunks_in))
                chunks_out = infer_write_chunks(op_node)
                for i, chunk in enumerate(chunks_out):
                    x[chunk] = x[op_node] + 1
                    y[chunk] = y[op_node] + i / 5
                added_chunks.append(len(chunks_out))

            max_y = max(added_chunks) / 5 + 1 
            current_height += max_y
            # set positions for next op nodes of the ops in this floor
            for op_node in queue:
                # Set position for next ops
                next_ops = infer_next_ops(op_node)
                next_ops_x = find_next_ops_x(x[op_node], len(next_ops))
                for i, next_op in enumerate(next_ops):
                    x[next_op] = next_ops_x[i]
                    if floor + 1 == g.vs["steps"][next_op]:
                        y[next_op] = y[op_node] + max_y
                        new_queue.append(next_op)
                    else:
                        to_set_y.append((op_node, next_op))
            
            i = 0
            while i < len(to_set_y):
                op_node, next_op = to_set_y[i][0], to_set_y[i][1]
                if floor + 1 == g.vs["steps"][next_op]:
                    y[next_op] = current_height
                    new_queue.append(next_op)
                    to_set_y.pop(i)
                    i -= 1
                i += 1

            queue = new_queue
            floor += 1
        eh = list(zip(x,y))
        return eh
                

                

    
    nnodes, operations, visited = assign_op_nodes(instruction_dag.operations)
    # nnodes -> total number of nodes in graph [initialized to the number of operations]
    # operations -> list with all the operations still to visit [initialized with the non 'st' operations]
    # visited -> set with the currently visited operations [initialized with the 'st' operations]


    # slot -> op 
    # Used when the condition  if chunk_index_src.issuperset(chunk_index_last_writer) is not respected
    non_root_chunks, nnodes = populate_non_root_chunks(collective, nnodes)

    visiting_order = []
    while len(operations) > 0:
        op = operations[0]

        if processed_dep(op):
            visited.add(op)
            visiting_order.append(op)
            nnodes = connect_generate_chunk(op, nnodes)    
            operations = operations[1:]

        else: operations = operations[1:] + [op]


    g = ig.Graph(nnodes, edges, directed=True)
    g.vs["type"] = vertex_types
    g.vs["labels"] = vertex_labels
    g.vs["colors"] = vertex_colors
    g.vs["shapes"] = vertex_shapes
    g.vs["types"] = vertex_types
    g.vs["steps"] = vertex_height
    
    g = remove_zero_degree_nodes(g)
    
    layout = "auto"
    layout = build_layout(g)
    style = {
    "vertex_label_size": 4,  
    "vertex_label_dist": 0,  
    "vertex_size": 10,  # Increase size
    "vertex_width": 17,
    "edge_width": 1,  
    "layout": layout,
    "vertex_color": g.vs["colors"],
    "vertex_label": g.vs["labels"],
    "vertex_shape": g.vs["shapes"],
    "bbox": (nnodes * 20, nnodes * 20)
    }
    ig.plot(g, **style, target="ciao.pdf")
    print(visiting_order)
    
         
