# Stable Fluid implementation in WebGPU
## Overview
This is my first attempt at writing code for the GPU. I implement the classic Stable Fluids algorithm based off Stam's paper: http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf

main branch has refactored code. I accidentally created the 'master' branch and was too lazy to delete it. The unfactored branch is before all the refactor commits.

## Setup
Project uses Bun and vite for setup. Install bun, run `bun install` then `bun dev`. Navigate to the port it mentions and you'll see crazy fluids.

## Benchmark
Run `bun run dev`, open `/bench/` on that server, and select a stroke, dye advection mode, diffusion count, and dye grid size. The demo uses limited MacCormack advection for dye by default. The benchmark runs scripted input on a empty simulation with a 256 × 256 velocity grid and reports CPU encoding, submission-to-completion latency, and final dye and velocity metrics. It waits for each frame to finish so timings are comparable across runs. Edge variation normalized by dye mass is a comparison metric for the same stroke and grid, not an image quality score. Use `bun run build` and `bun run preview` to measure a production build.

## Customization
Use the controls panel to change the color scheme, brush size, or dye resolution. Changing dye resolution restarts the simulation. The palette shader is in `src/Renderer.ts`.

## Disclaimer
Primary goal was to create a fluid as quickly as possible. The code is very bad. My next iteration will port this to Rust, where I will be much more conscious during development. Stay tuned!!

