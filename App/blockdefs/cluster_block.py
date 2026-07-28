def get_definition():
    return {
        "name": "ClusterBlock",
        "displayName": "Cluster",
        "category": "Analysis",
        "color": [0.83, 0.52, 0.21],
        "inputs": [
            {"name": "x", "type": "numeric", "required": True,
             "description": "X values"},
            {"name": "y", "type": "numeric", "required": True,
             "description": "Y values"},
        ],
        "outputs": [
            {"name": "labels", "type": "numeric",
             "description": "Cluster assignment per point"},
        ],
        "runner": "runCluster",
        "defaultParameters": {
            "n_clusters": 2,
        },
        "parameterDefinitions": [
            {"name": "n_clusters", "displayName": "Clusters", "type": "numeric"},
        ],
        "isCluster": True,
        "showPortLabels": True,
    }
