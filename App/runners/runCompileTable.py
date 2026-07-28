"""Runner for CompileTable block — collects input lists into a table structure.

Each connected input becomes a column. The variable name (port display name
or upstream port name) is used as the column name, and the index number is
the pulse number (row).
"""

import numpy as np


def run(inputs, params, block):
    columns = {}

    for port in block.input_ports:
        # Get data for this port — try both the port name and "addInput"
        val = inputs.get(port.name)
        if val is None:
            continue

        # Determine column name from the upstream wire's source port
        col_name = port.display_name
        if col_name == "Add input" and port.connections:
            wire = port.connections[0]
            if wire.source_port:
                col_name = wire.source_port.display_name or wire.source_port.name

        # Skip if still just a placeholder name with no data
        if col_name == "Add input":
            continue

        # Convert to flat list
        if isinstance(val, np.ndarray):
            col_data = val.ravel().tolist()
        elif isinstance(val, (list, tuple)):
            col_data = list(val)
        elif isinstance(val, (int, float)):
            col_data = [val]
        else:
            col_data = [val]

        columns[col_name] = col_data

    if not columns:
        return {"table": {}}

    # Pad shorter columns with None so all columns have equal length
    max_len = max(len(v) for v in columns.values())
    for key in columns:
        diff = max_len - len(columns[key])
        if diff > 0:
            columns[key] = columns[key] + [None] * diff

    # Add pulse number column (1-based index)
    table = {"Pulse": list(range(1, max_len + 1))}
    table.update(columns)

    # Show the viewer dialog (sortable / filterable / CSV export).
    try:
        from processing.compile_table_view import show_compile_table
        show_compile_table(table)
    except Exception:
        # Headless or import error — don't break the pipeline.
        pass

    return {"table": table}
