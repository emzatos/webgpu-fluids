# Stable Fluid implementation in WebGPU
## Overview
This is my first attempt at writing code for the GPU. I implement the classic Stable Fluids algorithm based off Stam's paper: http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf

main branch has refactored code. I accidentally created the 'master' branch and was too lazy to delete it. The unfactored branch is before all the refactor commits.

## Setup
Project uses Bun and vite for setup. Install bun, run `bun install` then `bun dev`. Navigate to the port it mentions and you'll see crazy fluids.

## Customization
There are some fun colors in `shaders.wgsl` which is the primary fragment shader. Just uncomment them and have a blast.

## Disclaimer
Primary goal was to create a fluid as quickly as possible. The code is very bad. My next iteration will port this to Rust, where I will be much more conscious during development. Stay tuned!!

