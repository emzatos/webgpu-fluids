struct SplatUniforms {
    position: vec2f,
    delta: vec2f,
    radius: f32,
    intensity: f32,
    grid_size: u32,
}

@group(0) @binding(0) var<uniform> uniforms: SplatUniforms;
@group(0) @binding(1) var density: texture_2d<f32>;
@group(0) @binding(2) var densityOutput: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var horizontal: texture_2d<f32>;
@group(0) @binding(4) var horizontalOutput: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var vertical: texture_2d<f32>;
@group(0) @binding(6) var verticalOutput: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn splat(@builtin(global_invocation_id) id: vec3u) {
    let cell = id.xy + vec2u(1);
    let size = uniforms.grid_size;
    if (any(cell > vec2u(size))) {
        return;
    }

    var dye = textureLoad(density, cell, 0).r;
    var velocity = vec2f(textureLoad(horizontal, cell, 0).r, textureLoad(vertical, cell, 0).r);
    let dist = distance(vec2f(cell), uniforms.position);
    if (dist < uniforms.radius) {
        let influence = (1.0 - dist / uniforms.radius) * uniforms.intensity;
        dye = (dye + influence) * 0.99;
        velocity = (velocity + uniforms.delta * influence) * 1.01;
    }

    writeField(densityOutput, cell, size, SCALAR_FIELD, dye);
    writeField(horizontalOutput, cell, size, HORIZONTAL_VELOCITY, velocity.x);
    writeField(verticalOutput, cell, size, VERTICAL_VELOCITY, velocity.y);
}
