import { d, std, tgpu, type TgpuBindGroup, type TgpuRoot } from "typegpu";

export type DyeTransport = "linear" | "maccormack";

const WORKGROUP_SIZE = 8;
const sampledField = { texture: d.texture2d(d.f32), sampleType: "float" as const };
const transportLayout = tgpu.bindGroupLayout({
  source: sampledField,
  output: { storageTexture: d.textureStorage2d("r32float") },
  horizontal: sampledField,
  vertical: sampledField,
});
const correctionLayout = tgpu.bindGroupLayout({ predicted: sampledField });

type TransportGroup = TgpuBindGroup<typeof transportLayout.entries>;
type CorrectionGroup = TgpuBindGroup<typeof correctionLayout.entries>;

const sourceRange = (position: d.v2f, size: d.v2u): d.v2f => {
  "use gpu";
  const lower = d.vec2i(std.floor(position));
  const upper = std.min(std.add(lower, d.vec2i(1)), d.vec2i(std.sub(size, d.vec2u(1))));
  const a = std.textureLoad(transportLayout.$.source, lower, 0).r;
  const b = std.textureLoad(transportLayout.$.source, d.vec2i(upper.x, lower.y), 0).r;
  const c = std.textureLoad(transportLayout.$.source, d.vec2i(lower.x, upper.y), 0).r;
  const e = std.textureLoad(transportLayout.$.source, upper, 0).r;
  return d.vec2f(std.min(std.min(a, b), std.min(c, e)), std.max(std.max(a, b), std.max(c, e)));
};

export class DyeAdvection {
  private readonly predict;
  private readonly correct;
  private readonly cache: {
    views: GPUTextureView[];
    prediction: TransportGroup;
    correction?: TransportGroup;
    predicted?: CorrectionGroup;
  }[] = [];

  constructor(private readonly root: TgpuRoot, dt: number, transport: DyeTransport) {
    const sampler = root.createSampler({ magFilter: "linear", minFilter: "linear" });
    const step = tgpu.const(d.f32, dt);

    const travel = (cell: d.v2f, size: d.v2f): d.v2f => {
      "use gpu";
      const interior = std.sub(size, d.vec2f(2));
      const world = std.div(std.sub(cell, d.vec2f(0.5)), interior);
      const velocitySize = d.vec2f(std.textureDimensions(transportLayout.$.horizontal));
      const velocityUV = std.div(
        std.add(std.mul(world, std.sub(velocitySize, d.vec2f(2))), d.vec2f(1)),
        velocitySize
      );
      const velocity = d.vec2f(
        std.textureSampleLevel(transportLayout.$.horizontal, sampler.$, velocityUV, 0).r,
        std.textureSampleLevel(transportLayout.$.vertical, sampler.$, velocityUV, 0).r
      );
      return std.mul(std.mul(velocity, interior), step.$);
    };
    const sampleUV = (position: d.v2f, size: d.v2f): d.v2f => {
      "use gpu";
      const clamped = std.clamp(position, d.vec2f(0.5), std.sub(size, d.vec2f(1.5)));
      return std.div(std.add(clamped, d.vec2f(0.5)), size);
    };

    const entry = tgpu.computeFn({
      in: { id: d.builtin.globalInvocationId },
      workgroupSize: [WORKGROUP_SIZE, WORKGROUP_SIZE],
    });
    this.predict = root.createComputePipeline({
      compute: entry(({ id }) => {
        "use gpu";
        const size = std.textureDimensions(transportLayout.$.source);
        if (std.any(std.ge(id.xy, size))) return;
        // Halo texels mirror the nearest interior cell.
        const cell = d.vec2f(std.clamp(id.xy, d.vec2u(1), std.sub(size, d.vec2u(2))));
        const sizeF = d.vec2f(size);
        const departure = std.sub(cell, travel(cell, sizeF));
        const dye = std.textureSampleLevel(
          transportLayout.$.source, sampler.$, sampleUV(departure, sizeF), 0
        ).r;
        std.textureStore(transportLayout.$.output, id.xy, d.vec4f(dye));
      }),
    });
    this.correct = transport === "maccormack" ? root.createComputePipeline({
      compute: entry(({ id }) => {
        "use gpu";
        const size = std.textureDimensions(transportLayout.$.source);
        if (std.any(std.ge(id.xy, size))) return;
        const last = std.sub(size, d.vec2u(2));
        const cell = std.clamp(id.xy, d.vec2u(1), last);
        const cellF = d.vec2f(cell);
        const sizeF = d.vec2f(size);
        const motion = travel(cellF, sizeF);
        const departure = std.sub(cellF, motion);
        const arrival = std.add(cellF, motion);
        const firstOrder = std.textureLoad(correctionLayout.$.predicted, cell, 0).r;
        let dye = firstOrder;
        const low = d.vec2f(1);
        const high = d.vec2f(last);
        const interiorPaths = std.all(std.ge(departure, low)) && std.all(std.le(departure, high))
          && std.all(std.ge(arrival, low)) && std.all(std.le(arrival, high));
        if (interiorPaths) {
          const reverse = std.textureSampleLevel(correctionLayout.$.predicted, sampler.$, sampleUV(arrival, sizeF), 0).r;
          const original = std.textureLoad(transportLayout.$.source, cell, 0).r;
          const range = sourceRange(departure, size);
          dye = std.clamp(firstOrder + 0.5 * (original - reverse), range.x, range.y);
        }
        std.textureStore(transportLayout.$.output, id.xy, d.vec4f(dye));
      }),
    }) : undefined;
  }

  advect(
    encoder: GPUCommandEncoder,
    dyeSize: number,
    source: GPUTextureView,
    predicted: GPUTextureView,
    output: GPUTextureView,
    horizontal: GPUTextureView,
    vertical: GPUTextureView,
    timestampWrites?: GPUComputePassTimestampWrites
  ) {
    const views = [source, predicted, output, horizontal, vertical];
    let bindings = this.cache.find((entry) => views.every((view, i) => view === entry.views[i]));
    if (!bindings) {
      bindings = {
        views,
        prediction: this.root.createBindGroup(transportLayout, {
          source, output: this.correct ? predicted : output, horizontal, vertical,
        }),
        correction: this.correct
          ? this.root.createBindGroup(transportLayout, { source, output, horizontal, vertical })
          : undefined,
        predicted: this.correct ? this.root.createBindGroup(correctionLayout, { predicted }) : undefined,
      };
      this.cache.push(bindings);
    }

    const groups = Math.ceil((dyeSize + 2) / WORKGROUP_SIZE);
    const pass = encoder.beginComputePass({ label: "dye advection", timestampWrites });
    this.predict.with(bindings.prediction).with(pass).dispatchWorkgroups(groups, groups);
    if (this.correct && bindings.correction && bindings.predicted) {
      this.correct.with(bindings.correction).with(bindings.predicted).with(pass).dispatchWorkgroups(groups, groups);
    }
    pass.end();
  }
}
