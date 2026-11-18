struct SplatUniforms {
    position: vec2f,
    delta: vec2f,
    radius: f32,
    intensity: f32,
    grid_size: u32,
    dye_size: u32,
}

@group(0) @binding(0) var<uniform> uniforms: SplatUniforms;
@group(0) @binding(1) var density: texture_2d<f32>;
@group(0) @binding(2) var densityOutput: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var horizontal: texture_2d<f32>;
@group(0) @binding(4) var horizontalOutput: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var vertical: texture_2d<f32>;
@group(0) @binding(6) var verticalOutput: texture_storage_2d<r32float, write>;

fn influence(position: vec2f) -> f32 {
    let distanceToBrush = distance(position, uniforms.position);
    if (distanceToBrush >= uniforms.radius) { return 0.0; }
    return 1.0 - distanceToBrush / uniforms.radius;
}

@compute @workgroup_size(8, 8)
fn splat(@builtin(global_invocation_id) id: vec3u) {
    let cell = id.xy + vec2u(1);
    let velocitySize = uniforms.grid_size;
    let dyeSize = uniforms.dye_size;

    if (all(cell <= vec2u(dyeSize))) {
        let velocityPosition = (vec2f(cell) - 0.5) * (f32(velocitySize) / f32(dyeSize)) + 0.5;
        let weight = influence(velocityPosition);
        var dye = textureLoad(density, cell, 0).r;
        if (weight > 0.0) { dye = (dye + weight * uniforms.intensity) * 0.99; }
        writeField(densityOutput, cell, dyeSize, SCALAR_FIELD, dye);
    }
    if (all(cell <= vec2u(velocitySize))) {
        let weight = influence(vec2f(cell));
        var velocity = vec2f(textureLoad(horizontal, cell, 0).r, textureLoad(vertical, cell, 0).r);
        if (weight > 0.0) { velocity = (velocity + uniforms.delta * (weight * uniforms.intensity)) * 1.01; }
        writeField(horizontalOutput, cell, velocitySize, HORIZONTAL_VELOCITY, velocity.x);
        writeField(verticalOutput, cell, velocitySize, VERTICAL_VELOCITY, velocity.y);
    }
}
