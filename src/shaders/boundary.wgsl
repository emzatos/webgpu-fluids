const X_STEP = vec2u(1, 0);
const Y_STEP = vec2u(0, 1);
const SCALAR_FIELD = 0u;
const HORIZONTAL_VELOCITY = 1u;
const VERTICAL_VELOCITY = 2u;

// The interior invocation owns its adjacent ghost cells, including corners.
fn writeField(
    output: texture_storage_2d<r32float, write>,
    cell: vec2u,
    size: u32,
    field: u32,
    value: f32,
) {
    textureStore(output, cell, vec4f(value));

    let left = cell.x == 1u;
    let right = cell.x == size;
    let bottom = cell.y == 1u;
    let top = cell.y == size;
    let farEdge = size + 1u;
    let xWall = vec4f(select(value, -value, field == HORIZONTAL_VELOCITY));
    let yWall = vec4f(select(value, -value, field == VERTICAL_VELOCITY));
    let corner = vec4f(select(0.0, value, field == SCALAR_FIELD));

    if (left) {
        textureStore(output, vec2u(0, cell.y), xWall);
        if (bottom) { textureStore(output, vec2u(0, 0), corner); }
        if (top) { textureStore(output, vec2u(0, farEdge), corner); }
    }
    if (right) {
        textureStore(output, vec2u(farEdge, cell.y), xWall);
        if (bottom) { textureStore(output, vec2u(farEdge, 0), corner); }
        if (top) { textureStore(output, vec2u(farEdge, farEdge), corner); }
    }
    if (bottom) { textureStore(output, vec2u(cell.x, 0), yWall); }
    if (top) { textureStore(output, vec2u(cell.x, farEdge), yWall); }
}
