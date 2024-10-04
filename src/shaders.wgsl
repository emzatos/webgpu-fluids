struct Uniforms {
    fun: f32,
    time: f32
}
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv : vec2f
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> uniforms : Uniforms;

@vertex fn vs (
    @builtin(vertex_index) vertexIndex : u32
) -> VertexOutput {
    let positions = array(
        vec2f(-1, 1),
        vec2f(-1, -1),
        vec2f(1, -1),
        vec2f(-1, 1),
        vec2f(1, 1),
        vec2f(1, -1)
    );

    let uvs = array(
        vec2f(0, 1),
        vec2f(0, 0),
        vec2f(1, 0),
        vec2f(0, 1),
        vec2f(1, 1),
        vec2f(1, 0),
    );
    var vout: VertexOutput;
    vout.position = vec4f(positions[vertexIndex], 0, 1);
    vout.uv = uvs[vertexIndex];
    return vout;
}

@fragment fn fs(vin: VertexOutput) -> @location(0) vec4f { 
    return uniforms.fun * textureSample(tex, samp, vec2f(vin.uv));
    
}