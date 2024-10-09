

struct Uniforms {
    diffusion_constant: f32,
    dt: f32,
    grid_size: u32
}

@group(0) @binding(0) var density_grid: texture_2d<f32>;
@group(0) @binding(1) var density_grid_write: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var prev_density_grid: texture_storage_2d<r32float, read>;
@group(0) @binding(3) var<uniform> uniforms : Uniforms;
@group(0) @binding(4) var velocity_grid: texture_storage_2d<rg32float, read>;
@group(0) @binding(5) var samp: sampler;

@compute @workgroup_size(8, 8) fn diffuse (
    @builtin(global_invocation_id) glob : vec3<u32>,
) {
    let x = glob.x;
    let y = glob.y;
    if (x == 0 || y == 0) {
        return;
    }
    if (x > uniforms.grid_size || y > uniforms.grid_size) {
        return;
    }

    let diffusion_rate = uniforms.dt * uniforms.diffusion_constant * f32(uniforms.grid_size);
    let inv = 1.0 / (1. + 4. * diffusion_rate);

    // diffuse step
    var value: f32 = 0.0;
    var neighborhood: f32 = 0.0;
    value += textureLoad(prev_density_grid, glob.xy).r;

    if (x > 0) { neighborhood += textureLoad(density_grid, vec2<u32>(glob.x-1, glob.y), 0).r; }
    if (y > 0) { neighborhood += textureLoad(density_grid, vec2<u32>(glob.x, glob.y-1), 0).r; }
    if (x <= uniforms.grid_size - 1) { neighborhood += textureLoad(density_grid, vec2<u32>(glob.x+1, glob.y), 0).r; }
    if (y <= uniforms.grid_size - 1) { neighborhood += textureLoad(density_grid, vec2<u32>(glob.x, glob.y+1), 0).r; }

    value = (value + diffusion_rate * neighborhood) * inv;
    textureStore(density_grid_write, glob.xy, vec4(value));
    setBoundary(0, density_grid_write, density_grid, glob.xy);
}

@compute @workgroup_size(8, 8) fn advect (
    @builtin(global_invocation_id) glob : vec3<u32>,
) {
    _ = prev_density_grid;
    let x = glob.x;
    let y = glob.y;
    if (x == 0 || y == 0) {
        return;
    }
    if (x > uniforms.grid_size || y > uniforms.grid_size ) {
        return;
    }

    let velocity = textureLoad(velocity_grid, glob.xy).rg;
    let dt = uniforms.dt;
    var pos = vec2f(f32(glob.x), f32(glob.y)) - velocity * dt * f32(uniforms.grid_size);
    pos = clamp(pos / f32(uniforms.grid_size), vec2f(0), vec2f(1));
    let sample = textureSampleLevel(density_grid, samp, pos, 0).r;
    textureStore(density_grid_write, glob.xy, vec4(sample, 0, 0, 1));
    setBoundary(0, density_grid_write, density_grid, glob.xy);
}

fn setBoundary(boundary: u32, write_grid: texture_storage_2d<r32float, write>, read_grid: texture_2d<f32>, pos: vec2<u32>)
{
    let x = pos.x;
    let y = pos.y;
    if (x == 0) {
      if (boundary != 1) {
        textureStore(write_grid, pos, textureLoad(read_grid, vec2<u32>(1, y), 0));
      } else {
        textureStore(write_grid, pos, vec4(-textureLoad(read_grid, vec2<u32>(1, y), 0).r, 0, 0, 1));
      }
    }

    if (x == uniforms.grid_size+1) {
      if (boundary != 1) {
        textureStore(write_grid, pos, textureLoad(read_grid, vec2<u32>(uniforms.grid_size, y), 0));
      } else {
        textureStore(write_grid, pos, vec4(-textureLoad(read_grid, vec2<u32>(uniforms.grid_size, y), 0).r, 0, 0, 1));
      }
    }

    if (y == 0) {
      if (boundary != 2) {
        textureStore(write_grid, pos, textureLoad(read_grid, vec2<u32>(x, 1), 0));
      } else {
        textureStore(write_grid, pos, vec4(0, -textureLoad(read_grid, vec2<u32>(x, 1), 0).r, 0, 1));
      }
    }

    if (y == uniforms.grid_size+1) {
      if (boundary != 2) {
        textureStore(write_grid, pos, textureLoad(read_grid, vec2<u32>(x, uniforms.grid_size), 0));
      } else {
        textureStore(write_grid, pos, vec4(0, -textureLoad(read_grid, vec2<u32>(x, uniforms.grid_size), 0).r, 0, 1));
      }
    }

    textureStore(write_grid, vec2u(0,0), 0.5 * (textureLoad(read_grid, vec2u(1,0), 0) + textureLoad(read_grid, vec2u(0,1), 0)));
    textureStore(write_grid, vec2u(0,uniforms.grid_size + 1), 0.5 * (textureLoad(read_grid, vec2u(1, uniforms.grid_size + 1), 0) + textureLoad(read_grid, vec2u(0, uniforms.grid_size), 0)));
    textureStore(write_grid, vec2u(uniforms.grid_size + 1, 0), 0.5 * (textureLoad(read_grid, vec2u(uniforms.grid_size,0), 0) + textureLoad(read_grid, vec2u(uniforms.grid_size + 1,1), 0)));
    textureStore(write_grid, vec2u(uniforms.grid_size + 1,uniforms.grid_size + 1), 0.5 * (textureLoad(read_grid, vec2u(uniforms.grid_size,uniforms.grid_size + 1), 0) + textureLoad(read_grid, vec2u(uniforms.grid_size + 1, uniforms.grid_size), 0)));

}
