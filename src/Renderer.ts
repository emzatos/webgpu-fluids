import { d, std, tgpu, type TgpuBindGroup, type TgpuCommandEncoder, type TgpuRoot } from "typegpu";

const densityLayout = tgpu.bindGroupLayout({
  density: { texture: d.texture2d(d.f32), sampleType: "float" },
});

const fullscreenVertex = tgpu.vertexFn({
  in: { vertexIndex: d.builtin.vertexIndex },
  out: { position: d.builtin.position, uv: d.vec2f },
})(({ vertexIndex }) => {
  "use gpu";
  const x = d.f32((vertexIndex << 1) & 2);
  const y = d.f32(vertexIndex & 2);
  return { position: d.vec4f(x * 2 - 1, y * 2 - 1, 0, 1), uv: d.vec2f(x, y) };
});

const rainbow = (density: number): d.v3f => {
  "use gpu";
  const sector = density * 6;
  const blend = 1 - std.abs((sector % 2) - 1);
  if (sector < 1) return d.vec3f(1, blend, 0);
  if (sector < 2) return d.vec3f(blend, 1, 0);
  if (sector < 3) return d.vec3f(0, 1, blend);
  if (sector < 4) return d.vec3f(0, blend, 1);
  if (sector < 5) return d.vec3f(blend, 0, 1);
  return d.vec3f(1, 0, blend);
};

const ember = (density: number): d.v3f => {
  "use gpu";
  const dye = std.max(density, 0);
  const intensity = std.sqrt(dye / (dye + 0.22));
  const background = d.vec3f(0.025, 0.035, 0.065);
  const orange = d.vec3f(0.93, 0.21, 0.07);
  const highlight = d.vec3f(1, 0.93, 0.69);
  if (intensity < 0.65) return std.mix(background, orange, intensity / 0.65);
  return std.mix(orange, highlight, (intensity - 0.65) / 0.35);
};

export class Renderer {
  private readonly pipeline;
  private readonly palette;
  private readonly densityGroups = new WeakMap<GPUTextureView, TgpuBindGroup<typeof densityLayout.entries>>();
  private activePalette = 0;

  constructor(private readonly root: TgpuRoot, private readonly context: GPUCanvasContext) {
    const sampler = root.createSampler({ magFilter: "linear", minFilter: "linear" });
    const palette = root.createUniform(d.u32, this.activePalette);
    this.palette = palette;

    const fragment = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
      "use gpu";
      const density = std.textureSample(densityLayout.$.density, sampler.$, uv).r;
      switch (palette.$) {
        case 0: return d.vec4f(rainbow(density), 1);
        case 1: return d.vec4f(d.vec3f(density), 1);
        case 2: return d.vec4f(d.vec3f(1 - density), 1);
        case 3: return d.vec4f(ember(density), 1);
        default: return d.vec4f(0, 0, 0, 1);
      }
    });
    this.pipeline = root.createRenderPipeline({
      vertex: fullscreenVertex,
      fragment,
      targets: { format: navigator.gpu.getPreferredCanvasFormat() },
    });
  }

  render(encoder: TgpuCommandEncoder, density: GPUTextureView, palette: number) {
    if (palette !== this.activePalette) {
      this.palette.write(palette);
      this.activePalette = palette;
    }
    let group = this.densityGroups.get(density);
    if (!group) {
      group = this.root.createBindGroup(densityLayout, { density });
      this.densityGroups.set(density, group);
    }
    const pass = encoder.beginRenderPass({ colorAttachments: { view: this.context } });
    this.pipeline.with(group).with(pass).draw(3);
    pass.end();
  }
}
