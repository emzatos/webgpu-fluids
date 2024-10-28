@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let tex_coords = vec2<i32>(id.xy);
    let color = textureLoad(input_texture, tex_coords, 0); // Read color from input texture

    let inverted_color = vec4<f32>(1.0 - color.rgb, color.a); // Invert the color
    textureStore(output_texture, tex_coords, inverted_color); // Write to output texture
}

