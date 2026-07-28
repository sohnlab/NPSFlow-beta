# Flow Control Blocks

## Start (`StartBlock`)
Pipeline entry point. Double-click to run the full pipeline.
- Outputs: `Run`
- Only one Start block per pipeline

## Sequence (`Priority`) — dynamic
Defines execution order for connected blocks. Outputs numbered 1, 2, 3, ...
- Inputs: `Run`
- Dynamic outputs: `Add output`
- Blocks connected to output 1 run before output 2, etc.

