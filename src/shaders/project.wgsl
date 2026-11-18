struct ProjectionUniforms {
    grid_size: u32,
}

@group(0) @binding(0) var<uniform> uniforms : ProjectionUniforms;

@group(0) @binding(1) var divergence_grid: texture_2d<f32>;
@group(0) @binding(2) var divergence_grid_write: texture_storage_2d<r32float, write>;

@group(0) @binding(3) var prev_pressure_grid: texture_2d<f32>;
@group(0) @binding(4) var pressure_grid: texture_2d<f32>;
@group(0) @binding(5) var pressure_grid_write: texture_storage_2d<r32float, write>;

@group(0) @binding(6) var horizontal_velocity_grid: texture_2d<f32>;
@group(0) @binding(7) var horizontal_velocity_grid_write: texture_storage_2d<r32float, write>;

@group(0) @binding(8) var vertical_velocity_grid: texture_2d<f32>;
@group(0) @binding(9) var vertical_velocity_grid_write: texture_storage_2d<r32float, write>;


@compute @workgroup_size(8, 8) fn update_divergence (
    @builtin(global_invocation_id) id : vec3<u32>,
) {

  // Compute divergence and warm-start the pressure solve.
  let cell = id.xy + vec2u(1);
  let size = uniforms.grid_size;
  if (any(cell > vec2u(size))) {
      return;
  }

  let horizontal_diff = textureLoad(horizontal_velocity_grid, cell + X_STEP, 0) - textureLoad(horizontal_velocity_grid, cell - X_STEP, 0);
  let vertical_diff = textureLoad(vertical_velocity_grid, cell + Y_STEP, 0) - textureLoad(vertical_velocity_grid, cell - Y_STEP, 0);
  let h = 1.0 / f32(size);

  let value = -0.5 * h * (horizontal_diff + vertical_diff);
  textureStore(divergence_grid_write, cell, value);
  textureStore(pressure_grid_write, cell, textureLoad(prev_pressure_grid, cell, 0) * 0.99);

}

@compute @workgroup_size(8, 8) fn update_pressure (
    @builtin(global_invocation_id) id : vec3<u32>,
) {
  // update pressure from divergence
  let cell = id.xy + vec2u(1);
  let size = uniforms.grid_size;
  if (any(cell > vec2u(size))) {
      return;
  }

  let div = textureLoad(divergence_grid, cell, 0);
  let adjacent_pressure = textureLoad(pressure_grid, cell - X_STEP, 0)
  + textureLoad(pressure_grid, cell + X_STEP, 0)
  + textureLoad(pressure_grid, cell - Y_STEP, 0)
  + textureLoad(pressure_grid, cell + Y_STEP, 0);

  textureStore(pressure_grid_write, cell, (div + adjacent_pressure) / 4.0);

}

@compute @workgroup_size(8, 8) fn apply_to_velocities (
    @builtin(global_invocation_id) id : vec3<u32>,
) {
  let cell = id.xy + vec2u(1);
  let size = uniforms.grid_size;
  if (any(cell > vec2u(size))) {
      return;
  }

  var horizontal_pressure = textureLoad(pressure_grid, cell + X_STEP, 0).r - textureLoad(pressure_grid, cell - X_STEP, 0).r;
  var vertical_pressure = textureLoad(pressure_grid, cell + Y_STEP, 0).r - textureLoad(pressure_grid, cell - Y_STEP, 0).r;
  let h = 1.0 / f32(size);
  horizontal_pressure = (horizontal_pressure * 0.5) / h;
  vertical_pressure = (vertical_pressure * 0.5) / h;

  let new_horizontal_velocity = textureLoad(horizontal_velocity_grid, cell, 0).r - horizontal_pressure;
  let new_vertical_velocity = textureLoad(vertical_velocity_grid, cell, 0).r - vertical_pressure;

  writeField(horizontal_velocity_grid_write, cell, size, HORIZONTAL_VELOCITY, new_horizontal_velocity);
  writeField(vertical_velocity_grid_write, cell, size, VERTICAL_VELOCITY, new_vertical_velocity);
}
