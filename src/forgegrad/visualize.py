from graphviz import Digraph


def trace(root):
    nodes, edges = set(), []

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.children:
                edges.append((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, format="svg", rankdir="LR"):
    """
    Visualizes the computation graph starting from `root`.
    rankdir: 'LR' (left-to-right) or 'TB' (top-to-bottom)
    """
    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        # Create a rectangular record node showing label and data
        label_str = (
            f"{{ {n.label} | data {n.data:.4f} | grad {n.grad:.4f} }}"
            if n.label
            else f"{{ data {n.data:.4f} }}"
        )
        dot.node(name=uid, label=label_str, shape="record")

        # If this node was created by an operation, create an op node and connect it
        if n._op:
            op_uid = uid + n._op
            dot.node(name=op_uid, label=n._op)
            dot.edge(op_uid, uid)

    for n1, n2 in edges:
        # Connect child node to the parent's operation node
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
