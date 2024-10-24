# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import igraph as ig
from msccl.language.ir import *
from msccl.language.rank_dag import *
from collections import deque

def visualize_chunk_dag(chunk_paths): # pragma: no cover
    frontier = []
    nnodes = 0
    vertex_label = []
    vertex_colors = []
    edges = []
    visited = set()

    def add_node(op, nnodes, vertex_label, vertex_colors):
        if op.num == -1:
            op.num = nnodes
            nnodes += 1
            if op.inst == ChunkInstruction.start:
                vertex_label.append(f'Start at {op.dst.rank}, {op.dst.index}.')
                vertex_colors.append('yellow')
            elif op.inst == ChunkInstruction.send:
                vertex_label.append(f'Send to Rank {op.dst.rank} {op.dst.index}. {op.steps_to_end}, {op.steps_from_start}')
                vertex_colors.append('blue')
            elif op.inst == ChunkInstruction.reduce:
                vertex_label.append(f'Reduce with {op.dst.rank} {op.dst.index}. {op.steps_to_end}, {op.steps_from_start}')
                vertex_colors.append('green')
        return nnodes

    for chunk, op in chunk_paths.items():
        if len(op.prev) == 0: 
            frontier.append(op)

    while len(frontier) > 0:
        op = frontier[0]
        if op in visited:
            frontier = frontier[1:]
        else:
            nnodes = add_node(op, nnodes, vertex_label, vertex_colors)
            for next_op in op.next:
                nnodes = add_node(next_op, nnodes, vertex_label, vertex_colors)
                edges.append([op.num, next_op.num])
            frontier = frontier[1:] + op.next
            visited.add(op)

    g = ig.Graph(nnodes, edges, directed=True)
    layout = g.layout(layout=ig.Graph.layout_grid)
    ig.plot(g, vertex_label=vertex_label, vertex_color=vertex_colors, layout='auto', target="out.png")

def visualize_rank_dag(operations): # pragma: no cover
    frontier = []
    nnodes = 0
    vertex_label = []
    vertex_colors = []
    edges = []
    visited = set()
    colors = ['red', 'green', 'blue', 'yellow', 'teal', 'pink', 'purple', 'orange']

    def add_node(op, nnodes, vertex_label, vertex_colors):
        if op.num == -1:
            op.num = nnodes
            nnodes += 1
            # Add new node to graph
            if op.inst == Instruction.start:
                vertex_label.append(f'Chunk {op.src.index} Rank {op.src.rank}')
            elif op.inst == Instruction.send:
                vertex_label.append(f'S to Rank {op.dst.rank}')
            elif op.inst == Instruction.recv:
                vertex_label.append(f'R from {op.src.rank}')
            elif op.inst == Instruction.recv_reduce_copy:
                vertex_label.append(f'RRC from {op.src.rank}')
            else:
                vertex_label.append(f'{op.inst}')

            # Add colors
            if op.inst == Instruction.start:
                vertex_colors.append('gray')
            else:
                vertex_colors.append(colors[op.tb % len(colors)])
        return nnodes

    for slot, op in operations.items():
        if len(op.prev) == 0: 
            frontier.append(op)

    while len(frontier) > 0:
        op = frontier[0]

        if op in visited:
            frontier = frontier[1:]
        else:
            nnodes = add_node(op, nnodes, vertex_label, vertex_colors)

        for next_op in op.next:
            nnodes = add_node(next_op, nnodes, vertex_label, vertex_colors)
            edges.append([op.num, next_op.num])
            frontier = frontier[1:] + list(op.next)
        visited.add(op)

    g = ig.Graph(nnodes, edges, directed=True)
    layout = g.layout(layout=ig.Graph.layout_grid)
    style = {
    "vertex_label_size": 4,  
    "vertex_label_dist": 1,  
    "vertex_size": 20, 
    "edge_width": 1.5,  
    "layout": 'rt',
    "vertex_color": vertex_colors,
    "vertex_label": vertex_label
    }
    ig.plot(g, **style, target="out.pdf")

import re
def visualize_instruction_dag(operations, num_ranks, chunk_factor, inplace): # pragma: no cover
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

    #formats the labels of chunks
    def format_label(buffer, rank, chunk):
        superscript = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
        subscript = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
        return f"{buffer}{str(rank).translate(superscript)}{str(chunk).translate(subscript)}"

    def _write(rank, buffer, index, size, node, read=False):
        prev_ops = set()
        for i in range(index, index+size):
            slot = (rank, buffer.value, i)

            # If there are active readers - these are the previous operations
            # Else the previous operation is the last write (if there is one)
            readers = last_readers[slot]
            if len(readers) > 0:
                prev_ops.update(readers)
            elif slot in last_writer:
                prev_ops.add(last_writer[slot])
  
            # Set the last_writer to this op, and clear all readers
            last_writer[slot] = node
            last_readers[slot] = []


    def _read(rank, buffer, index, size, node):
        prev_ops = set()
        for i in range(index, index+size):
            slot = (rank, buffer.value, i)
            assert slot in last_writer, f"Slot has never been written before a read-type {op}"
            # The previous operation for a reader is the last write to the slot
            writer = last_writer[slot]
            prev_ops.add(writer)
            last_readers[slot].append(node)

    def build_layout(g):
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
        for v in range(len(vertex_types)):
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
        normalized_layout = [((x - min_x) / (max_x - min_x) * canvas_scale_factor, 
                            (y - min_y) / (max_y - min_y) * canvas_scale_factor) for x, y in layout]
        return normalized_layout
    
    def add_chunk_node(rank, index, buffer, edge=None):
        nnodes+=1
        vertex_label.append(f"C_{rank}_{index}_{buffer}")
        vertex_colors.append("yellow")
        vertex_shapes.append("rectangle")
        vertex_types.append("c")
        if edge:
            edges.append(edge)
        return nnodes
    
    def add_send_node():
        nnodes+=1
        vertex_label.append("s")
        vertex_colors.append(instruction_color_mapping[op.inst])
        vertex_shapes.append("circle")
        vertex_types.append("op")

    def add_recv_node():
        nnodes+=1
        vertex_label.append("r")
        vertex_colors.append(instruction_color_mapping[op.inst])
        vertex_shapes.append("circle")
        vertex_types.append("op")



    def add_node(op:Op, nnodes, prev=-1, next=-1):
        # The start operation has basically no infos
        if op.inst == Instruction.start: return nnodes

        nnodes += 1
        if str(op.rank) not in initialized_ranks: # If there's no entry point for the rank yet
            if (not inplace and last_writer[(op.src.rank, op.src.buffer.value, op.src.index)] == 'i') or (inplace and last_writer[(op.src.rank, op.src.buffer.value, op.src.index)] == 'i'): # if the chunk has never been written
                roots.append(nnodes)
                initialized_ranks.add(str(op.rank))
                # Add the src chunk as entrypoint
                _write(op.rank, op.src.buffer, op.src.index, op.src.size, nnodes)
                vertex_label.append(f"C_{op.src.rank}_{op.src.index}_{op.src.buffer}")
                vertex_colors.append("yellow")
                vertex_shapes.append("rectangle")
                vertex_types.append("c")

                # Add the operation
                nnodes += 1
                op.num = nnodes
                vertex_label.append(op.inst)
                vertex_colors.append(instruction_color_mapping[op.inst])
                vertex_shapes.append("circle")
                vertex_types.append("op")

                # Add the dst chunk
                nnodes += 1
                _write(op.rank, op.dst.buffer, op.dst.index, op.dst.size, nnodes)
                vertex_label.append(f"C_{op.dst.rank}_{op.dst.index}_{op.dst.buffer}")
                vertex_colors.append("cyan")
                vertex_shapes.append("rectangle")
                edges.append([nnodes-2, nnodes-1])
                edges.append([nnodes, nnodes-1])
                vertex_types.append("c")

        
        elif op.inst == Instruction.send: #send is a read operation so doesn't write chunks
            op.num = nnodes
            vertex_label.append(op.inst)
            vertex_colors.append(instruction_color_mapping[op.inst])
            vertex_shapes.append("circle")
            vertex_types.append("op")
        
        else: 
            if op.is_recv():
                op.num = nnodes
                vertex_label.append(op.inst)
                vertex_colors.append(instruction_color_mapping[op.inst])
                vertex_shapes.append("circle")
                vertex_types.append("op")
                # Add an edge from the sender
                edges.append([op.send_match.num, op.num])
                #Add chunk being written
                nnodes += 1
                edges.append([nnodes, op.num])
                vertex_label.append(f"C_{op.send_match.dst.rank}_{op.send_match.dst.index}_{op.send_match.dst.buffer}")
                vertex_colors.append("cyan")
                vertex_shapes.append("rectangle")
                vertex_types.append("c")

            
            else:
                op.num = nnodes
                vertex_label.append(op.inst)
                vertex_colors.append(instruction_color_mapping[op.inst])
                vertex_shapes.append("circle")
                vertex_types.append("op")
                #Add chunk being written
                nnodes += 1
                edges.append([nnodes, op.num])
                vertex_label.append(f"C_{op.dst.rank}_{op.dst.index}_{op.dst.buffer}")
                vertex_colors.append("cyan")
                vertex_shapes.append("rectangle")
                vertex_types.append("c")

        if prev!=-1:
            edges.append([prev, op.num])
        if next!=-1:
            edges.append([op.num, next])

        return nnodes
    
    last_writer = {(rank, buffer, index):"i" for rank in range(num_ranks) for index in range(num_ranks*chunk_factor) for buffer in ["i", "o"]}
    last_readers = {(rank, buffer, index):[] for rank in range(num_ranks) for index in range(num_ranks*chunk_factor) for buffer in ["i", "o"]}
    initialized_ranks = set()
    visited = set()

    roots = []
    vertex_label = []
    vertex_colors = []
    vertex_shapes = []
    edges = []
    vertex_types = []
    

    nnodes = -1
    
    for slot, ops in operations.items():
        frontier = [ops]
        while len(frontier) > 0:
            op = frontier[0]
            if op not in visited:
                nnodes = add_node(op, nnodes)
                visited.add(op)
            for next_op in op.next:
                if next_op not in visited:
                    nnodes = add_node(next_op, nnodes, prev=op.num)
                    visited.add(next_op)

            frontier = frontier[1:] + op.next

    g = ig.Graph(nnodes, edges, directed=False)
    g.vs["type"] = vertex_types
    layout = build_layout(g)

    

    g = ig.Graph(nnodes, edges, directed=True)
    from math import log10
    style = {
    "vertex_label_size": 4,  
    "vertex_label_dist": 0,  
    "vertex_size": 10,  # Increase size
    "vertex_width": 17,
    "edge_width": 1,  
    "layout": layout,
    "vertex_color": vertex_colors,
    "vertex_label": vertex_label,
    "vertex_shape": vertex_shapes,
    "bbox": (nnodes * 10, nnodes * 10)
    }
    ig.plot(g, **style, target="ciao.pdf")
    

    # for slot, ops in operations.items():
    #     print(slot)

    # print("\n")

    # for label in vertex_label:
    #     print(re.search(r'Ref\(([^()]+)\).*?Ref', label).group(1))


def visualize_instruction_ir(program: Program):
    class Gpu_repr:
        rank:int
        id:int

    gpu_vertex = []
    nnodes = 0
    num_gpu = len(program.gpus)
    for gpu in program.gpus:
        gpu_vertex.add(gpu.rank)
