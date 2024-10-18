

struct Uniforms {
    diffusion_constant: f32,
    dt: f32,
    grid_size: u32,
    boundaryValue: u32,
    h: f32
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@group(0) @binding(1) var divergence_grid: texture_2d<f32>;
@group(0) @binding(2) var divergence_grid_write: texture_storage_2d<r32float, write>;

@group(0) @binding(3) var prev_pressure_grid: texture_2d<f32>;
@group(0) @binding(4) var pressure_grid: texture_2d<f32>;
@group(0) @binding(5) var pressure_grid_write: texture_storage_2d<r32float, write>;

@group(0) @binding(6) var horizontal_velocity_grid: texture_2d<f32>;
@group(0) @binding(7) var horizontal_velocity_grid_write: texture_storage_2d<r32float, write>;

@group(0) @binding(8) var vertical_velocity_grid: texture_2d<f32>;
@group(0) @binding(9) var vertical_velocity_grid_write: texture_storage_2d<r32float, write>;


@compute @workgroup_size(8, 8) fn project_begin (
    @builtin(global_invocation_id) glob : vec3<u32>,
) {
  _ = divergence_grid;
  _ = pressure_grid;
  _ = horizontal_velocity_grid_write;
  _ = vertical_velocity_grid_write;

  // update divergence, neutralize pressure
  let x = glob.x;
  let y = glob.y;
  if (x == 0 || y == 0) {
      return;
  }
  if (x > uniforms.grid_size || y > uniforms.grid_size) {
      return;
  }

  let horizontal_diff = textureLoad(horizontal_velocity_grid, vec2u(glob.x+1, glob.y), 0) - textureLoad(horizontal_velocity_grid, vec2u(glob.x-1, glob.y), 0);
  let vertical_diff = textureLoad(vertical_velocity_grid, vec2u(glob.x, glob.y+1), 0) - textureLoad(vertical_velocity_grid, vec2u(glob.x, glob.y-1), 0);

  let value = -0.5 * uniforms.h * (horizontal_diff + vertical_diff);
  textureStore(divergence_grid_write, glob.xy, value);
  textureStore(pressure_grid_write, glob.xy, textureLoad(prev_pressure_grid, glob.xy, 0) * 0.99);

  //set bnd
}

@compute @workgroup_size(8, 8) fn project_middle (
    @builtin(global_invocation_id) glob : vec3<u32>,
) {
  _ = divergence_grid_write;
  _ = vertical_velocity_grid;
  _ = vertical_velocity_grid_write;
  _ = horizontal_velocity_grid;
  _ = horizontal_velocity_grid_write;
  _ = prev_pressure_grid;
  // update pressure from divergence
  let x = glob.x;
  let y = glob.y;
  if (x == 0 || y == 0) {
      return;
  }
  if (x > uniforms.grid_size || y > uniforms.grid_size) {
      return;
  }

  let div = textureLoad(divergence_grid, glob.xy, 0);
  let adjacent_pressure = textureLoad(pressure_grid, vec2u(glob.x-1, glob.y), 0) 
  + textureLoad(pressure_grid, vec2u(glob.x+1, glob.y), 0)
  + textureLoad(pressure_grid, vec2u(glob.x, glob.y-1), 0)
  + textureLoad(pressure_grid, vec2u(glob.x, glob.y+1), 0);

  textureStore(pressure_grid_write, glob.xy, (div + adjacent_pressure) / 4.0);

}

@compute @workgroup_size(8, 8) fn project_end (
    @builtin(global_invocation_id) glob : vec3<u32>,
) {
  _ = divergence_grid_write;
  _ = divergence_grid;
  _ = pressure_grid_write;
  _ = prev_pressure_grid;
  let x = glob.x;
  let y = glob.y;
  if (x == 0 || y == 0) {
      return;
  }
  if (x > uniforms.grid_size || y > uniforms.grid_size) {
      return;
  }
  // apply to velocities
  var horizontal_pressure = textureLoad(pressure_grid, vec2u(glob.x + 1, glob.y), 0) - textureLoad(pressure_grid, vec2u(glob.x - 1, glob.y), 0);
  var vertical_pressure = textureLoad(pressure_grid, vec2u(glob.x, glob.y+1), 0) - textureLoad(pressure_grid, vec2u(glob.x, glob.y-1), 0);

  horizontal_pressure = (horizontal_pressure * 0.5) / uniforms.h;
  vertical_pressure = (vertical_pressure * 0.5) / uniforms.h;

  let new_horizontal_velocity = textureLoad(horizontal_velocity_grid, glob.xy, 0) - horizontal_pressure;
  let new_vertical_velocity = textureLoad(vertical_velocity_grid, glob.xy, 0) - vertical_pressure;

  textureStore(horizontal_velocity_grid_write, glob.xy, new_horizontal_velocity);
  textureStore(vertical_velocity_grid_write, glob.xy, new_vertical_velocity);
}


