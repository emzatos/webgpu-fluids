struct DiffusionUniforms {
    diffusion_constant: f32,
    dt: f32,
    grid_size: u32,
    boundaryValue: u32,
}

@group(0) @binding(0) var<uniform> uniforms: DiffusionUniforms;
@group(0) @binding(1) var estimate: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var source: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn diffuse(@builtin(global_invocation_id) id: vec3u) {
    let cell = id.xy + vec2u(1);
    let size = uniforms.grid_size;
    if (any(cell > vec2u(size))) {
        return;
    }

    let rate = uniforms.dt * uniforms.diffusion_constant * f32(size);
    let neighbors = textureLoad(estimate, cell - X_STEP, 0).r
                  + textureLoad(estimate, cell + X_STEP, 0).r
                  + textureLoad(estimate, cell - Y_STEP, 0).r
                  + textureLoad(estimate, cell + Y_STEP, 0).r;
    let value = (textureLoad(source, cell, 0).r + rate * neighbors) / (1.0 + 4.0 * rate);
    writeField(output, cell, size, uniforms.boundaryValue, value);
}
