import igraph as ig

# Create a new graph
g = ig.Graph(directed=True)

# Add vertices
vertices = ["MSCCL-IR", "Program", "GPU 0", "GPU 3", "GPU 5", 
            "Threadblock 0", "Threadblock 1", "Threadblock 2"]
g.add_vertices(vertices)

# Add edges
edges = [("MSCCL-IR", "Program"),
         ("Program", "GPU 0"), ("Program", "GPU 3"), ("Program", "GPU 5"),
         ("GPU 3", "Threadblock 0"), ("GPU 3", "Threadblock 1"), ("GPU 3", "Threadblock 2")]
g.add_edges(edges)

# Set vertex labels
g.vs["label"] = vertices

# Set vertex styles
g.vs["shape"] = ["none"] + ["rectangle"] * (len(vertices) - 1)
g.vs["size"] = [20 if v != "MSCCL-IR" else 1 for v in vertices]

# Create a layout
layout = g.layout("tree")

# Create a larger plotting area
visual_style = {}
visual_style["vertex_size"] = 40
visual_style["vertex_label_size"] = 12
visual_style["edge_curved"] = 0.1
visual_style["layout"] = layout
visual_style["bbox"] = (800, 600)
visual_style["margin"] = 40

# Add connection and instruction information
connection_info = {
    "Threadblock 0": "Connections\nsend peer: 5\nreceive peer: 4\nchannel: 0",
    "Threadblock 1": "Connections\nsend peer: 0\nreceive peer: 1\nchannel: 0",
    "Threadblock 2": "Connections\nsend peer: 5\nreceive peer: 4\nchannel: 1"
}

instruction_info = {
    "Threadblock 0": "Instructions\n0: send('in', 4)\n1: rcs('in', 2)\n2: rrc('in', 0)",
    "Threadblock 1": "Instructions\n0: send('in', 0)\n   dep=(tb0, 2)\n1: rrcs('in', 1)\n2: recv('in', 0)",
    "Threadblock 2": "Instructions\n0: send('in', 0)\n   dep=(tb1, 2)\n1: rcs('in', 2)\n2: r('in', 4)"
}

# Add text labels for connections and instructions
for v in g.vs:
    if v["name"] in connection_info:
        g.vs[v.index]["connection_info"] = connection_info[v["name"]]
        g.vs[v.index]["instruction_info"] = instruction_info[v["name"]]

# Custom drawing function to add text labels
def add_vertex_labels(visual_style, context, coords, vertex):
    if "connection_info" in vertex.attributes():
        context.set_font_size(8)
        context.set_source_rgb(0, 0, 0)
        x, y = coords[vertex.index]
        lines = vertex["connection_info"].split('\n')
        for i, line in enumerate(lines):
            context.move_to(x + 50, y - 40 + i * 10)
            context.show_text(line)
        
        lines = vertex["instruction_info"].split('\n')
        for i, line in enumerate(lines):
            context.move_to(x + 50, y + 20 + i * 10)
            context.show_text(line)

# Plot the graph
ig.plot(g, target="msccl_ir_graph.pdf", **visual_style)

print("Graph has been saved as 'msccl_ir_graph.pdf'")