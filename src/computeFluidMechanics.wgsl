

struct Uniforms {
    diffusion_constant: f32,
    dt: f32,
    grid_size: u32
}

@group(0) @binding(0) var density_grid: texture_storage_2d<r32float, read>;
@group(0) @binding(1) var density_grid_write: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var prev_density_grid: texture_storage_2d<r32float, read>;
@group(0) @binding(3) var<uniform> uniforms : Uniforms;

@compute @workgroup_size(8, 8) fn diffuse (
    @builtin(global_invocation_id) glob : vec3<u32>,
) {
    let x = glob.x;
    let y = glob.y;
    if (x >= uniforms.grid_size || y >= uniforms.grid_size) {
        return;
    }

    let diffusion_rate = uniforms.dt * uniforms.diffusion_constant * f32(uniforms.grid_size);
    let inv = 1.0 / (1. + 4. * diffusion_rate);

    // diffuse step
    var value: f32 = 0.0;
    var neighborhood: f32 = 0.0;
    value += textureLoad(prev_density_grid, glob.xy).r;

    if (x > 0) { neighborhood += textureLoad(density_grid, vec2<u32>(glob.x-1, glob.y)).r; }
    if (y > 0) { neighborhood += textureLoad(density_grid, vec2<u32>(glob.x, glob.y-1)).r; }
    if (x < uniforms.grid_size - 1) { neighborhood += textureLoad(density_grid, vec2<u32>(glob.x+1, glob.y)).r; } 
    if (y < uniforms.grid_size - 1) { neighborhood += textureLoad(density_grid, vec2<u32>(glob.x, glob.y+1)).r; }
    

    value = (value + diffusion_rate * neighborhood) * inv;
    textureStore(density_grid_write, glob.xy, vec4(value));
}

