import { DensityTextures } from "./DensityTextures";
import {
  ComputePipelines, divergenceLayout, pressureLayout, projectionLayout,
} from "./ComputePipelines";
import { d, type TgpuBindGroup, type TgpuRoot } from "typegpu";

export class FluidSimulation {
  private device: GPUDevice;
  private textureSize: number;
  private densities: DensityTextures;
  private velX: DensityTextures;
  private velY: DensityTextures;
  private divergence: DensityTextures;
  private pressure: DensityTextures;
  private computePipelines: ComputePipelines;
  private sampler: GPUSampler;
  private readonly projectionGroups = new Map<
    GPUComputePipeline, { views: readonly GPUTextureView[]; group: GPUBindGroup }[]
  >();

  constructor(private readonly root: TgpuRoot, textureSize: number) {
    const { device } = root;
    this.device = device;
    this.textureSize = textureSize;
    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });
    var [velXData, velYData] = this.initializeVelocity(textureSize);
    var zeroData = new Float32Array({
      length: (textureSize + 2) * (textureSize + 2),
    }).fill(0.0);

    this.densities = new DensityTextures({
      device,
      textureSize: textureSize + 2,
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
    splatPass.dispatchWorkgroups(this.textureSize / 8, this.textureSize / 8);
    splatPass.end();

    encoder.copyTextureToTexture(
      {
        texture: this.densities.writeTexture,
      },
      {
        texture: this.densities.prevTexture,
      },
      {
        width: this.textureSize,
        height: this.textureSize,
      }
    );
    encoder.copyTextureToTexture(
      {
        texture: this.velX.writeTexture,
      },
      {
        texture: this.velX.prevTexture,
      },
      {
        width: this.textureSize,
        height: this.textureSize,
      }
    );
    encoder.copyTextureToTexture(
      {
        texture: this.velY.writeTexture,
      },
      {
        texture: this.velY.prevTexture,
      },
      {
        width: this.textureSize,
        height: this.textureSize,
      }
    );

    this.densities.flip();
    this.velX.flip();
    this.velY.flip();
    this.densities.swap();
    this.velX.swap();
    this.velY.swap();
  }

  private velocityUpdate(encoder: GPUCommandEncoder) {
    this.velX.swap();
    this.velY.swap();
    this.performDiffusion(encoder, this.velX, this.computePipelines.velocityDiffusionUniforms);
    this.performDiffusion(encoder, this.velY, this.computePipelines.velocityDiffusionUniforms);
    this.performProjection(encoder);
    this.velX.swap();
    this.velY.swap();
    this.performAdvection(
      encoder,
      this.velX,
      this.velX.prevTexture,
      this.velY.prevTexture
    );
    this.performAdvection(
      encoder,
      this.velY,
      this.velX.prevTexture,
      this.velY.prevTexture
    );
    this.performProjection(encoder);
  }

  private densityUpdate(encoder: GPUCommandEncoder) {
    this.densities.swap();
    this.performDiffusion(encoder, this.densities, this.computePipelines.densityDiffusionUniforms);
    this.densities.swap();
    this.performAdvection(
      encoder,
      this.densities,
      this.velX.readTexture,
      this.velY.readTexture
    );
    this.densities.flip();
  }

  private performDiffusion(
    encoder: GPUCommandEncoder,
    texture: DensityTextures,
    uniforms: ComputePipelines["densityDiffusionUniforms"]
  ) {
    const uniformBuffer = this.root.unwrap(uniforms);
    for (var i = 0; i < 20; i++) {
      const diffusionPass = encoder.beginComputePass(
        this.computePipelines.diffusionData.descriptor
      );

      const diffusionBindGroup = this.device.createBindGroup({
        layout:
          this.computePipelines.diffusionData.pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: { buffer: uniformBuffer },
          },
          {
            binding: 1,
            resource: texture.readView,
          },
          {
            binding: 2,
            resource: texture.writeView,
          },
          {
            binding: 3,
            resource: texture.prevView,
          },
        ],
      });

      diffusionPass.setPipeline(this.computePipelines.diffusionData.pipeline);
      diffusionPass.setBindGroup(0, diffusionBindGroup);
      diffusionPass.dispatchWorkgroups(
        this.textureSize / 8,
        this.textureSize / 8
      );
      diffusionPass.end();
      texture.flip();
    }
  }

  private performAdvection(
    encoder: GPUCommandEncoder,
    texture: DensityTextures,
    horizontal: GPUTexture,
    vertical: GPUTexture
  ) {
    const advectPass = encoder.beginComputePass(
      this.computePipelines.advectData.descriptor
    );
    const advectBindGroup = this.device.createBindGroup({
      layout: this.computePipelines.advectData.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: this.root.unwrap(this.computePipelines.advectData.uniforms) },
        },
        {
          binding: 1,
          resource: texture.readView,
        },
        {
          binding: 2,
          resource: texture.writeView,
        },
        {
          binding: 3,
          resource: texture.prevView, //unused
        },
        {
          binding: 4,
          resource: horizontal.createView(),
        },
        {
          binding: 5,
          resource: vertical.createView(),
        },
        {
          binding: 6,
          resource: this.sampler,
        },
      ],
    });

    advectPass.setPipeline(this.computePipelines.advectData.pipeline);
    advectPass.setBindGroup(0, advectBindGroup);
    advectPass.dispatchWorkgroups(this.textureSize / 8, this.textureSize / 8);
    advectPass.end();
    texture.flip();
  }

  private performProjection(encoder: GPUCommandEncoder) {
    const { divergence, pressure, velX, velY } = this;
    const { divergencePipeline, pressurePipeline, projectionPipeline, projectionUniforms } =
      this.computePipelines;
    const divergenceGroup = this.projectionGroup(
      divergencePipeline,
      [divergence.writeView, pressure.prevView, pressure.writeView, velX.readView, velY.readView],
      ([divergenceOutput, previousPressure, pressureOutput, horizontal, vertical]) =>
        this.root.createBindGroup(divergenceLayout, {
          uniforms: projectionUniforms, divergenceOutput, previousPressure,
          pressureOutput, horizontal, vertical,
        })
    );
    this.dispatchProjection(encoder, divergencePipeline, divergenceGroup, "divergence");
    pressure.flip();
    pressure.swap();
    divergence.flip();

    const pressureGroup = (input: GPUTextureView, output: GPUTextureView) =>
      this.projectionGroup(
        pressurePipeline,
        [divergence.readView, input, output],
        ([divergence, pressure, pressureOutput]) =>
          this.root.createBindGroup(pressureLayout, {
            uniforms: projectionUniforms, divergence, pressure, pressureOutput,
          })
      );
    const forward = pressureGroup(pressure.readView, pressure.writeView);
    const backward = pressureGroup(pressure.writeView, pressure.readView);
    for (let i = 0; i < 200; i += 2) {
      this.dispatchProjection(encoder, pressurePipeline, forward, "pressure");
      this.dispatchProjection(encoder, pressurePipeline, backward, "pressure");
    }

    const projectionGroup = this.projectionGroup(
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

  private projectionGroup<T extends GPUTextureView[]>(
    pipeline: GPUComputePipeline,
    views: T,
    create: (views: T) => TgpuBindGroup
  ): GPUBindGroup {
    let groups = this.projectionGroups.get(pipeline);
    if (!groups) {
      groups = [];
      this.projectionGroups.set(pipeline, groups);
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
    pass.dispatchWorkgroups(this.textureSize / 8, this.textureSize / 8);
    pass.end();
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
