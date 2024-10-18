
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv : vec2f
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> time : f32;  // Add time as a uniform

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

// @fragment fn fs(vin: VertexOutput) -> @location(0) vec4f { 
//     _ = time;
//     let value = textureSample(tex, samp, vec2f(vin.uv)).r;
//     //rainbow
//    // var color: vec3<f32> = hsl_to_rgb(value * 360.0, 1.0, 0.5);
//    // black background with white fluid
//    // var color: vec3<f32> = vec3<f32>(value, value, value);
//    // white background with black fluid
//   //  var color: vec3<f32> = vec3<f32>(1.0 - value, 1.0 - value, 1.0 - value);
//   // very cool
//    //var color: vec3<f32> = hsl_to_rgb((value) * 360.0, 2.0, 60);
//     let angle = atan2(vin.uv.y - 0.5, vin.uv.x - 0.5);
// let sheen = 0.5 + 0.5 * sin(angle * 10.0 + time + value * 5.0);
// var color = vec3<f32>(0.6, 0.6, 0.6) + sheen * vec3<f32>(0.2, 0.3, 0.3);
//     return vec4<f32>(color, 1.0);    
// }

fn hash(p: vec2f) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn noise(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash(i), hash(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y
    );
}

@fragment fn fs(vin: VertexOutput) -> @location(0) vec4f {
    _ = time;
    let value = textureSample(tex, samp, vin.uv).r;
    let laplacian = fwidth(value) * 0.5;
    let reaction = sin(value * 10.0 + time) * 0.5 + 0.5;
    let mixed = clamp(value + laplacian * 50.0 - reaction * 0.1, 0.0, 1.0);
    var color = vec3<f32>(
        mixed,
        mixed * mixed,
        1.0 - mixed * mixed
    );
    return vec4<f32>(color, 1.0);
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    if (s == 0.0) {
        return vec3<f32>(l, l, l);
    }

    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0));
    let m = l - c / 2.0;

    var r: f32;
    var g: f32;
    var b: f32;

    if (h < 60.0) {
        r = c; g = x; b = 0.0;
    } else if (h < 120.0) {
        r = x; g = c; b = 0.0;
    } else if (h < 180.0) {
        r = 0.0; g = c; b = x;
    } else if (h < 240.0) {
        r = 0.0; g = x; b = c;
    } else if (h < 300.0) {
        r = x; g = 0.0; b = c;
    } else {
        r = c; g = 0.0; b = x;
    }

    return vec3<f32>(r + m, g + m, b + m);
}