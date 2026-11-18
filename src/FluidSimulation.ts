import { DensityTextures } from "./DensityTextures";
import {
  ComputePipelines, diffusionLayout, divergenceLayout, pressureLayout, projectionLayout,
} from "./ComputePipelines";
import { d, type TgpuBindGroup, type TgpuRoot } from "typegpu";

const WORKGROUP_SIZE = 8;
const DIFFUSION_ITERATIONS = 20;
const PRESSURE_ITERATIONS = 200;

export class FluidSimulation {
  private device: GPUDevice;
  private readonly workgroups: number;
  private densities: DensityTextures;
  private velX: DensityTextures;
  private velY: DensityTextures;
  private divergence: DensityTextures;
  private pressure: DensityTextures;
  private computePipelines: ComputePipelines;
  private sampler: GPUSampler;
  private readonly computeGroups = new Map<
    GPUComputePipeline, { views: readonly GPUTextureView[]; group: GPUBindGroup }[]
  >();

  constructor(
    private readonly root: TgpuRoot,
    private readonly textureSize: number,
    { startEmpty = false }: { startEmpty?: boolean } = {}
  ) {
    const { device } = root;
    this.device = device;
    this.workgroups = Math.ceil(textureSize / WORKGROUP_SIZE);
    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });
    const zeroData = new Float32Array((textureSize + 2) ** 2);
    const [velXData, velYData] = startEmpty
      ? [zeroData, zeroData]
      : this.initializeVelocity(textureSize);

    this.densities = new DensityTextures({
      device,
      textureSize: textureSize + 2,
      data: startEmpty ? zeroData : undefined,
    });

    this.velX = new DensityTextures({
      device,
      textureSize: textureSize + 2,
      data: velXData,
    });

    this.velY = new DensityTextures({
      device,
      textureSize: textureSize + 2,
      data: velYData,
    });

    this.divergence = new DensityTextures({
      device,
      textureSize: textureSize + 2,
      data: zeroData,
    });

    this.pressure = new DensityTextures({
      device,
      textureSize: textureSize + 2,
      data: zeroData,
    });

    this.computePipelines = new ComputePipelines(root, textureSize);
  }

  private initializeVelocity = (
    textureSize: number
  ): [Float32Array, Float32Array] => {
    const N = textureSize;
    const velX = new Float32Array((N + 2) * (N + 2));
    const velY = new Float32Array((N + 2) * (N + 2));

    for (let x = 1; x <= N; ++x) {
      for (let y = 1; y <= N; ++y) {
        const dx = x - (N + 2) / 2;
        const dy = y - (N + 2) / 2;
        const dist = Math.hypot(dx, dy);
        if (dist < 20) {
          velX[y * (N + 2) + x] = dy * 0.15;
          velY[y * (N + 2) + x] = -dx * 0.15;
        }
      }
    }

    return [velX, velY];
  };

  public update(encoder: GPUCommandEncoder) {
    this.velocityUpdate(encoder);
    this.densityUpdate(encoder);
  }

  public splat(
    encoder: GPUCommandEncoder,
    x: number,
    y: number,
    dx: number,
    dy: number,
    radius: number,
    intensity: number
  ) {
    const splatPass = encoder.beginComputePass(
      this.computePipelines.splatData.descriptor
    );
    this.computePipelines.splatData.uniforms.write({
      pos: d.vec2f(x, y),
      delta: d.vec2f(dx, dy),
      radius: radius,
      intensity: intensity,
      grid_size: this.textureSize,
    });
    const splatBindGroup = this.device.createBindGroup({
      layout: this.computePipelines.splatData.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: this.root.unwrap(this.computePipelines.splatData.uniforms) },
        },
        {
          binding: 1,
          resource: this.densities.readView,
        },
        {
          binding: 2,
          resource: this.densities.writeView,
        },
        {
          binding: 3,
          resource: this.velX.readView,
        },
        {
          binding: 4,
          resource: this.velX.writeView,
        },
        {
          binding: 5,
          resource: this.velY.readView,
        },
        {
          binding: 6,
          resource: this.velY.writeView,
        },
      ],
    });

    splatPass.setPipeline(this.computePipelines.splatData.pipeline);
    splatPass.setBindGroup(0, splatBindGroup);
    splatPass.dispatchWorkgroups(this.workgroups, this.workgroups);
    splatPass.end();

    this.densities.flip();
    this.velX.flip();
    this.velY.flip();
  }

  private velocityUpdate(encoder: GPUCommandEncoder) {
    const { horizontal, vertical } = this.computePipelines.fieldUniforms;
    this.performDiffusion(encoder, this.velX, horizontal.diffusion);
    this.performDiffusion(encoder, this.velY, vertical.diffusion);
    this.performProjection(encoder);

    const horizontalSource = this.velX.readView;
    const verticalSource = this.velY.readView;
    this.performAdvection(encoder, this.velX, horizontalSource, verticalSource, horizontal.advection);
    this.performAdvection(encoder, this.velY, horizontalSource, verticalSource, vertical.advection);
    this.performProjection(encoder);
  }

  private densityUpdate(encoder: GPUCommandEncoder) {
    const { density } = this.computePipelines.fieldUniforms;
    this.performDiffusion(encoder, this.densities, density.diffusion);
    this.performAdvection(encoder, this.densities, this.velX.readView, this.velY.readView, density.advection);
  }

  private performDiffusion(
    encoder: GPUCommandEncoder,
    field: DensityTextures,
    uniforms: ComputePipelines["fieldUniforms"]["density"]["diffusion"]
  ) {
    const { pipeline, descriptor } = this.computePipelines.diffusionData;
    const source = field.readView;
    const firstOutput = field.writeView;
    const secondOutput = field.prevView;
    const groupFor = (estimate: GPUTextureView, output: GPUTextureView) =>
      this.computeGroup(pipeline, [estimate, output, source], ([estimate, output, source]) =>
        this.root.createBindGroup(diffusionLayout, { uniforms, estimate, output, source })
      );
    const firstGroup = groupFor(source, firstOutput);
    const groups = [groupFor(secondOutput, firstOutput), groupFor(firstOutput, secondOutput)];

    const pass = encoder.beginComputePass(descriptor);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, firstGroup);
    pass.dispatchWorkgroups(this.workgroups, this.workgroups);
    for (let i = 1; i < DIFFUSION_ITERATIONS; i++) {
      pass.setBindGroup(0, groups[i % 2]);
      pass.dispatchWorkgroups(this.workgroups, this.workgroups);
    }
    pass.end();
    if (DIFFUSION_ITERATIONS % 2 === 0) field.swapWriteAndScratch();
    field.flip();
  }

  private performAdvection(
    encoder: GPUCommandEncoder,
    field: DensityTextures,
    horizontal: GPUTextureView,
    vertical: GPUTextureView,
    uniforms: ComputePipelines["fieldUniforms"]["density"]["advection"]
  ) {
    const { pipeline, descriptor } = this.computePipelines.advectData;
    const group = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.root.unwrap(uniforms) } },
        { binding: 1, resource: field.readView },
        { binding: 2, resource: field.writeView },
        { binding: 3, resource: horizontal },
        { binding: 4, resource: vertical },
        { binding: 5, resource: this.sampler },
      ],
    });
    const pass = encoder.beginComputePass(descriptor);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(this.workgroups, this.workgroups);
    pass.end();
    field.flip();
  }

  private performProjection(encoder: GPUCommandEncoder) {
    const { divergence, pressure, velX, velY } = this;
    const { divergencePipeline, pressurePipeline, projectionPipeline, projectionUniforms } =
      this.computePipelines;
    const divergenceGroup = this.computeGroup(
      divergencePipeline,
      [divergence.writeView, pressure.readView, pressure.writeView, velX.readView, velY.readView],
      ([divergenceOutput, previousPressure, pressureOutput, horizontal, vertical]) =>
        this.root.createBindGroup(divergenceLayout, {
          uniforms: projectionUniforms, divergenceOutput, previousPressure,
          pressureOutput, horizontal, vertical,
        })
    );
    this.dispatchProjection(encoder, divergencePipeline, divergenceGroup, "divergence");
    pressure.flip();
    divergence.flip();

    const pressureGroup = (input: GPUTextureView, output: GPUTextureView) =>
      this.computeGroup(
        pressurePipeline,
        [divergence.readView, input, output],
        ([divergence, pressure, pressureOutput]) =>
          this.root.createBindGroup(pressureLayout, {
            uniforms: projectionUniforms, divergence, pressure, pressureOutput,
          })
      );
    const groups = [
      pressureGroup(pressure.readView, pressure.writeView),
      pressureGroup(pressure.writeView, pressure.readView),
    ];
    const pressurePass = encoder.beginComputePass({ label: "pressure" });
    pressurePass.setPipeline(pressurePipeline);
    for (let i = 0; i < PRESSURE_ITERATIONS; i++) {
      pressurePass.setBindGroup(0, groups[i % 2]);
      pressurePass.dispatchWorkgroups(this.workgroups, this.workgroups);
    }
    pressurePass.end();
    if (PRESSURE_ITERATIONS % 2 !== 0) pressure.flip();

    const projectionGroup = this.computeGroup(
      projectionPipeline,
      [pressure.readView, velX.readView, velX.writeView, velY.readView, velY.writeView],
      ([pressure, horizontal, horizontalOutput, vertical, verticalOutput]) =>
        this.root.createBindGroup(projectionLayout, {
          uniforms: projectionUniforms, pressure, horizontal, horizontalOutput,
          vertical, verticalOutput,
        })
    );
    this.dispatchProjection(encoder, projectionPipeline, projectionGroup, "project velocity");
    velX.flip();
    velY.flip();
  }

  private computeGroup<T extends GPUTextureView[]>(
    pipeline: GPUComputePipeline,
    views: T,
    create: (views: T) => TgpuBindGroup
  ): GPUBindGroup {
    let groups = this.computeGroups.get(pipeline);
    if (!groups) {
      groups = [];
      this.computeGroups.set(pipeline, groups);
    }
    const cached = groups.find((entry) => views.every((view, i) => view === entry.views[i]));
    if (cached) return cached.group;

    const group = this.root.unwrap(create(views));
    groups.push({ views, group });
    return group;
  }

  private dispatchProjection(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    group: GPUBindGroup,
    label: string
  ) {
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(this.workgroups, this.workgroups);
    pass.end();
  }

  public getDensityView() {
    return this.densities.readView;
  }

  public getDensityTexture() {
    return this.densities.readTexture;
  }

  public getVelXTexture() {
    return this.velX.readTexture;
  }

  public getVelYTexture() {
    return this.velY.readTexture;
  }
}
