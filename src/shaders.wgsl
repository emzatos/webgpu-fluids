
@vertex fn vs (
    @builtin(vertex_index) vertexIndex : u32
) -> @builtin(position) vec4f {
    return vec4(1.0, 0.0, 0.0, 1.0);
}

@fragment fn fs() -> @location(0) vec4f { 
    return vec4(1.0, 0.0, 0.0, 1.0);
}