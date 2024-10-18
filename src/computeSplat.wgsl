// computeSplat.wgsl

// Define the workgroup size
struct SplatUniforms {
    position: vec2<f32>,
    delta: vec2<f32>,
    radius: f32,
    intensity: f32,
};

@group(0) @binding(0) var<uniform> Uniforms : SplatUniforms;
@group(0) @binding(1) var density_grid: texture_2d<f32>;
@group(0) @binding(2) var density_grid_write: texture_storage_2d<r32float, write>;

@group(0) @binding(3) var horizontal_velocity_grid: texture_2d<f32>;
@group(0) @binding(4) var horizontal_velocity_grid_write: texture_storage_2d<r32float, write>;

@group(0) @binding(5) var vertical_velocity_grid: texture_2d<f32>;
@group(0) @binding(6) var vertical_velocity_grid_write: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pos = vec2<f32>(global_id.xy);
    let mousePos = Uniforms.position;
    let delta = Uniforms.delta;
    let radius = Uniforms.radius;
    let intensity = Uniforms.intensity;

    let dist = distance(pos, mousePos);
    if (dist < radius) {
        let influence = (1.0 - dist / radius) * intensity;
        
        // Update velocity texture
        let horizontalVelocity = textureLoad(horizontal_velocity_grid, global_id.xy, 0).x;
        let verticalVelocity = textureLoad(vertical_velocity_grid, global_id.xy, 0).x;
        let velocity = vec2f(horizontalVelocity, verticalVelocity);
        var newVelocity = velocity + delta * influence;
        newVelocity *= 1.01;
        textureStore(horizontal_velocity_grid_write, global_id.xy, vec4<f32>(newVelocity.x));
        textureStore(vertical_velocity_grid_write, global_id.xy, vec4<f32>(newVelocity.y));
        // Update density texture
        let density = textureLoad(density_grid, global_id.xy, 0).x;
        var newDensity = density + influence;
        newDensity *= 0.99;
        textureStore(density_grid_write, global_id.xy, vec4<f32>(newDensity, 0.0, 0.0, 1.0));
    }
}