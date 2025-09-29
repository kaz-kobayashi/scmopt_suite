def visualize_mermaid(model: Model, rankdir: str = "LR"):
    """Generate Mermaid diagram definitions for a supply chain model.

    This function creates two Mermaid diagram definitions:
    1. A main graph showing the structure of the supply chain, including:
       - Nodes (facilities/locations)
       - Activities (production processes)
       - Products
       - Resources
       - Modes (different ways to perform activities)
       - Requirements and relationships between components
    
    2. A Bill of Materials (BOM) graph showing the product structure and component relationships

    Parameters
    ----------
    model : Model
        The supply chain model containing nodes, activities, resources, and their relationships
    rankdir : str, optional
        Direction of the graph layout ("LR" for left-to-right, "TB" for top-to-bottom), by default "LR"

    Returns
    -------
    tuple[str, str]
        A tuple containing two strings:
        - First string: Mermaid definition for the main supply chain graph
        - Second string: Mermaid definition for the BOM graph

    Notes
    -----
    The diagrams use different node styles to represent different components:
    - Activities: Red rectangles
    - Products: Yellow ovals
    - Modes: Blue 3D boxes
    - Resources: Green trapeziums
    - Nodes: Light blue containers
    """
    def draw_activity(a: str, act: Activity, nodes: list, edges: list, subgraph_name: str):
        """Draw an activity node and its related components in the Mermaid diagram.

        Parameters
        ----------
        a : str
            Activity identifier
        act : Activity
            Activity object containing product and mode information
        nodes : list
            List to store node definitions
        edges : list
            List to store edge definitions
        subgraph_name : str
            Name of the parent subgraph
        """
        # Activity node
        nodes.append(f"{a}[{a}]:::activity")
        # Product node
        product_id = f"{a}_product"
        nodes.append(f"{product_id}(({a}\\n{act.product.name})):::product")
        
        for m, mode in act.modes.items():
            mode_id = f"{mode.name}_{a}"
            nodes.append(f"{mode_id}[{mode.name}]:::mode")
            edges.append(f"{a} -.-> {mode_id}")
            
            if mode.variable_requirement is not None:
                requirement = mode.variable_requirement
            elif mode.fixed_requirement is not None:
                requirement = mode.fixed_requirement
            else:
                requirement = {}
                
            for rname, _ in requirement.items():
                edges.append(f"{mode_id} --> {rname}")

    def generate_graph():
        """Generate the main supply chain graph in Mermaid format.

        Returns
        -------
        str
            Mermaid diagram definition string for the main graph
        """
        nodes = []
        edges = []
        subgraphs = []
        act_in_node = set()

        # Style definitions
        mermaid = ["graph " + rankdir,
                  "classDef activity fill:#f00,stroke:#f00",
                  "classDef product fill:#ff0,stroke:#ff0",
                  "classDef mode fill:#00f,stroke:#00f",
                  "classDef resource fill:#0f0,stroke:#0f0",
                  "classDef node fill:#lightblue,stroke:#000"]

        # Handle nodes and their activities
        if model.nodes is not None:
            for i, node in model.nodes.items():
                node_subgraph = [f"subgraph {i}[Node {i}]"]
                nodes.append(f"node_{i}[{i}]:::node")
                
                if node.activities is not None:
                    for a, act in node.activities.items():
                        act_in_node.add(a)
                        draw_activity(a, act, nodes, edges, i)
                
                node_subgraph.extend(nodes)
                node_subgraph.append("end")
                subgraphs.append("\n".join(node_subgraph))

        # Handle arcs
        if model.arcs is not None:
            for (i, j), arc in model.arcs.items():
                label = ""
                if arc.activities is not None:
                    label = " |" + ",".join(arc.activities.keys()) + "|"
                edges.append(f"node_{i} -->{label} node_{j}")

        # Handle resources
        if model.resources is not None:
            for r, _ in model.resources.items():
                nodes.append(f"{r}[{r}]:::resource")

        # Handle activities not in nodes
        if model.activities is not None:
            for a, act in model.activities.items():
                if a not in act_in_node:
                    draw_activity(a, act, nodes, edges, "main")

        # Combine all elements
        mermaid.extend(subgraphs)
        mermaid.extend(nodes)
        mermaid.extend(edges)
        
        return "\n".join(mermaid)

    def generate_bom():
        """Generate the Bill of Materials (BOM) graph in Mermaid format.

        Returns
        -------
        str
            Mermaid diagram definition string for the BOM graph
        """
        nodes = []
        edges = []
        seen_products = set()
        
        mermaid = ["graph " + rankdir]
        
        if model.activities is not None:
            for a, act in model.activities.items():
                product_name = act.product.name
                if product_name not in seen_products:
                    nodes.append(f"{product_name}(({product_name})):::product")
                    seen_products.add(product_name)
                
                for m, mode in act.modes.items():
                    if mode.components is not None:
                        for child, weight in mode.components.items():
                            if child not in seen_products:
                                nodes.append(f"{child}(({child})):::product")
                                seen_products.add(child)
                            
                            label = f" |{weight}|" if weight != 1.0 else ""
                            edges.append(f"{child} -->{label} {product_name}")

        mermaid.extend(nodes)
        mermaid.extend(edges)
        mermaid.append("classDef product fill:#fff,stroke:#000")
        
        return "\n".join(mermaid)

    return generate_graph(), generate_bom()
