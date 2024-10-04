

struct Uniforms {
    fun: f32,
    time: f32
}

@group(0) @binding(0) var tex: texture_storage_2d<r32float, write>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> uniforms : Uniforms;

@compute @workgroup_size(8, 8) fn compute (
    @builtin(global_invocation_id) glob : vec3<u32>,
) {
    _ = samp;
    let value = vec4f(sin(f32(glob.x) * uniforms.fun + uniforms.time) * cos(0.2 * uniforms.fun + uniforms.time * f32(glob.y)));
    textureStore(tex, glob.xy, value);
}

