// computeSplat.wgsl

// Define the workgroup size
struct SplatUniforms {
    position: vec2<f32>,
    delta: vec2<f32>,
    radius: f32,
    intensity: f32,
    grid_size: u32,
};

@group(0) @binding(0) var<uniform> uniforms : SplatUniforms;
@group(0) @binding(1) var density_grid: texture_2d<f32>;
@group(0) @binding(2) var density_grid_write: texture_storage_2d<r32float, write>;

@group(0) @binding(3) var horizontal_velocity_grid: texture_2d<f32>;
@group(0) @binding(4) var horizontal_velocity_grid_write: texture_storage_2d<r32float, write>;

@group(0) @binding(5) var vertical_velocity_grid: texture_2d<f32>;
@group(0) @binding(6) var vertical_velocity_grid_write: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (uniforms.radius == 0) {
        return;
      }
      if (!(all(id.xy >= vec2u(1)) && all(id.xy <= vec2u(uniforms.grid_size)))) {
        return;
      }

      var density = textureLoad(density_grid, id.xy, 0).x;
      var velocityX = textureLoad(horizontal_velocity_grid, id.xy, 0).x;
      var velocityY = textureLoad(vertical_velocity_grid, id.xy, 0).x;
      var velocity = vec2<f32>(velocityX, velocityY);

      let dx = vec2f(id.xy) - uniforms.position;
      let dist = length(dx);
      if (dist < uniforms.radius) {
        let f = 1.0 - dist / uniforms.radius;
        // let f = 1.0;
        
        density += f * uniforms.intensity;
        velocity += f * uniforms.delta;
      }
      
      density *= 0.99;
      
      textureStore(density_grid_write, id.xy, vec4(density, 0, 0, 0));
      textureStore(horizontal_velocity_grid_write, id.xy, vec4(velocity.x, 0, 0, 0));
      textureStore(vertical_velocity_grid_write, id.xy, vec4(velocity.y, 0, 0, 0));
    }