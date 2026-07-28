# Variable Blocks

## Data Bus (`DataBus`) — dynamic
Named data bus for passing data between disconnected parts of the pipeline.
- Fixed inputs: `Run`, dynamic: `Add input`
- Fixed outputs: `Run`, dynamic mirrors inputs
- Parameter: `pool` (default: "Reference")
- Display name updates to show pool name

## Load Geometry (`LoadGeometry`) — dynamic
Loads saved device geometry from Reference store.
- Parameter: `refVariable` (default: "geometry")

## Sample Rate (`GlobalSampleRate`)
Publishes the effective sample rate to all downstream blocks.
- Inputs: `sample rate` (base, optional), `DS factor` (downsample N, optional)
- Computes `effective = base / N`; publishes globals `sampleRate` and
  `downsampleFactor`.
- Base falls back to the current global then 10 kHz when unwired; N defaults to 1.
- Runs after Preprocess when its `DS factor` input is wired from the Preprocess
  block's `DSfactor` output, so downstream blocks see the reduced rate.
