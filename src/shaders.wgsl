
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv : vec2f
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

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
    let value = 1.1 * textureSample(tex, samp, vec2f(vin.uv));
    return pow(value.xxxw, vec4(2.5, 1.7, 0.6, 1));
    
}