

struct AdvectUniforms {
    dt: f32,
    grid_size: u32,
    boundaryValue: u32
}

@group(0) @binding(0) var<uniform> uniforms : AdvectUniforms;

@group(0) @binding(1) var density_grid: texture_2d<f32>;
@group(0) @binding(2) var density_grid_write: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var prev_density_grid: texture_storage_2d<r32float, read>;

@group(0) @binding(4) var velocity_grid_horizontal: texture_2d<f32>;
@group(0) @binding(5) var velocity_grid_vertical: texture_2d<f32>;

@group(0) @binding(6) var samp: sampler;

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

    let velocity_x = textureLoad(velocity_grid_horizontal, glob.xy, 0).r;
    let velocity_y = textureLoad(velocity_grid_vertical, glob.xy, 0).r;
    let velocity = vec2f(velocity_x, velocity_y);

    let dt0 = vec2f(uniforms.dt * f32(uniforms.grid_size));
    let backPos = clamp(
      vec2f(glob.xy) - dt0 * velocity,
      vec2f(0.5),
      vec2f(f32(uniforms.grid_size) + vec2f(0.5))
    );
    let texSize = vec2f(textureDimensions(density_grid));
    let samplePos = (backPos + 0.5) / texSize;
    let value = textureSampleLevel(density_grid, samp, samplePos, 0);
    textureStore(density_grid_write, glob.xy, value);
    setBoundary(density_grid_write, density_grid, glob.xy);
}

fn setBoundary(write_grid: texture_storage_2d<r32float, write>, read_grid: texture_2d<f32>, pos: vec2<u32>)
{
    let x = pos.x;
    let y = pos.y;
    if (x == 0) {
      if (uniforms.boundaryValue != 1) {
        textureStore(write_grid, pos, textureLoad(read_grid, vec2<u32>(1, y), 0));
      } else {
        textureStore(write_grid, pos, vec4(-textureLoad(read_grid, vec2<u32>(1, y), 0).r, 0, 0, 1));
      }
    }

    if (x == uniforms.grid_size+1) {
      if (uniforms.boundaryValue != 1) {
        textureStore(write_grid, pos, textureLoad(read_grid, vec2<u32>(uniforms.grid_size, y), 0));
      } else {
        textureStore(write_grid, pos, vec4(-textureLoad(read_grid, vec2<u32>(uniforms.grid_size, y), 0).r, 0, 0, 1));
      }
    }

    if (y == 0) {
      if (uniforms.boundaryValue != 2) {
        textureStore(write_grid, pos, textureLoad(read_grid, vec2<u32>(x, 1), 0));
      } else {
        textureStore(write_grid, pos, vec4(0, -textureLoad(read_grid, vec2<u32>(x, 1), 0).r, 0, 1));
      }
    }

    if (y == uniforms.grid_size+1) {
      if (uniforms.boundaryValue != 2) {
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
