struct AdvectUniforms {
    dt: f32,
    grid_size: u32,
    boundaryValue: u32,
}

@group(0) @binding(0) var<uniform> uniforms: AdvectUniforms;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var horizontal: texture_2d<f32>;
@group(0) @binding(4) var vertical: texture_2d<f32>;
@group(0) @binding(5) var linearSampler: sampler;

@compute @workgroup_size(8, 8)
fn advect(@builtin(global_invocation_id) id: vec3u) {
    let cell = id.xy + vec2u(1);
    let size = uniforms.grid_size;
    if (any(cell > vec2u(size))) {
        return;
    }

    let velocity = vec2f(textureLoad(horizontal, cell, 0).r, textureLoad(vertical, cell, 0).r);
    let backPos = clamp(
        vec2f(cell) - (uniforms.dt * f32(size)) * velocity,
        vec2f(0.5),
        vec2f(f32(size) + 0.5),
    );
    let uv = (backPos + 0.5) / vec2f(textureDimensions(source));
    let value = textureSampleLevel(source, linearSampler, uv, 0).r;
    writeField(output, cell, size, uniforms.boundaryValue, value);
}
