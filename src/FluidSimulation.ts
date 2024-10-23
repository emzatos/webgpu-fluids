import { DensityTextures } from "./DensityTextures";
import { ComputePipelines } from "./ComputePipelines";
import { vec2f } from "typegpu/data/index";

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

  constructor(device: GPUDevice, textureSize: number) {
    this.device = device;
    this.textureSize = textureSize;
    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });
    var [velXData, velYData] = this.initializeVelocity(textureSize);

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
    });

    this.pressure = new DensityTextures({
      device,
      textureSize: textureSize + 2,
    });

    this.computePipelines = new ComputePipelines(device);
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
      pos: vec2f(x, y),
      delta: vec2f(dx, dy),
      radius: radius,
      intensity: intensity,
      grid_size: this.textureSize,
    });
    const splatBindGroup = this.device.createBindGroup({
      layout: this.computePipelines.splatData.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.computePipelines.splatData.uniforms,
        },
        {
          binding: 1,
          resource: this.densities.readTexture.createView(),
        },
        {
          binding: 2,
          resource: this.densities.writeTexture.createView(),
        },
        {
          binding: 3,
          resource: this.velX.readTexture.createView(),
        },
        {
          binding: 4,
          resource: this.velX.writeTexture.createView(),
        },
        {
          binding: 5,
          resource: this.velY.readTexture.createView(),
        },
        {
          binding: 6,
          resource: this.velY.writeTexture.createView(),
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
    this.performDiffusion(encoder, this.velX, 0.0001, 0.002);
    this.performDiffusion(encoder, this.velY, 0.0001, 0.002);
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
    this.performDiffusion(encoder, this.densities, 0.0001, 0.02);
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
    diffusionConstant: number,
    dt: number
  ) {
    this.computePipelines.diffusionData.uniforms.write({
      diffusion_constant: diffusionConstant,
      dt: dt,
      grid_size: this.textureSize,
      boundaryValue: 2,
      h: 1.0 / this.textureSize,
    });

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
            resource: this.computePipelines.diffusionData.uniforms,
          },
          {
            binding: 1,
            resource: texture.readTexture.createView(),
          },
          {
            binding: 2,
            resource: texture.writeTexture.createView(),
          },
          {
            binding: 3,
            resource: texture.prevTexture.createView(),
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
          resource: this.computePipelines.advectData.uniforms,
        },
        {
          binding: 1,
          resource: texture.readTexture.createView(),
        },
        {
          binding: 2,
          resource: texture.writeTexture.createView(),
        },
        {
          binding: 3,
          resource: texture.prevTexture.createView(), //unused
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
    this.computeDivergence(
      encoder,
      this.divergence,
      this.pressure,
      this.velX,
      this.velY
    );
    this.computePressure(
      encoder,
      this.divergence,
      this.pressure,
      this.velX,
      this.velY
    );
    this.projectOnVelocity(
      encoder,
      this.divergence,
      this.pressure,
      this.velX,
      this.velY
    );
  }

  private computeDivergence(
    encoder: GPUCommandEncoder,
    div: DensityTextures,
    pres: DensityTextures,
    horiz: DensityTextures,
    vert: DensityTextures
  ) {
    const divergencePass = encoder.beginComputePass(
      this.computePipelines.projectStageOneData.descriptor
    );
    const divergenceBindGroup = this.createProjectionBindGroup(
      this.computePipelines.projectStageOneData.pipeline,
      div,
      pres,
      horiz,
      vert
    );
    divergencePass.setPipeline(
      this.computePipelines.projectStageOneData.pipeline
    );
    divergencePass.setBindGroup(0, divergenceBindGroup);
    divergencePass.dispatchWorkgroups(
      this.textureSize / 8,
      this.textureSize / 8
    );
    divergencePass.end();
    pres.flip();
    pres.swap();
    div.flip();
  }

  private computePressure(
    encoder: GPUCommandEncoder,
    div: DensityTextures,
    pres: DensityTextures,
    horiz: DensityTextures,
    vert: DensityTextures
  ) {
    for (var i = 0; i < 200; i++) {
      const pressurePass = encoder.beginComputePass(
        this.computePipelines.projectStageTwoData.descriptor
      );
      const pressureBindGroup = this.createProjectionBindGroup(
        this.computePipelines.projectStageTwoData.pipeline,
        div,
        pres,
        horiz,
        vert
      );
      pressurePass.setPipeline(
        this.computePipelines.projectStageTwoData.pipeline
      );
      pressurePass.setBindGroup(0, pressureBindGroup);
      pressurePass.dispatchWorkgroups(
        this.textureSize / 8,
        this.textureSize / 8
      );
      pressurePass.end();
      pres.flip();
    }
  }

  private projectOnVelocity(
    encoder: GPUCommandEncoder,
    div: DensityTextures,
    pres: DensityTextures,
    horiz: DensityTextures,
    vert: DensityTextures
  ) {
    const projectionPass = encoder.beginComputePass(
      this.computePipelines.projectStageThreeData.descriptor
    );
    const projectionBindGroup = this.createProjectionBindGroup(
      this.computePipelines.projectStageThreeData.pipeline,
      div,
      pres,
      horiz,
      vert
    );
    projectionPass.setPipeline(
      this.computePipelines.projectStageThreeData.pipeline
    );
    projectionPass.setBindGroup(0, projectionBindGroup);
    projectionPass.dispatchWorkgroups(
      this.textureSize / 8,
      this.textureSize / 8
    );
    projectionPass.end();
    horiz.flip();
    vert.flip();
  }

  private createProjectionBindGroup(
    pipeline: GPUComputePipeline,
    div: DensityTextures,
    pres: DensityTextures,
    horiz: DensityTextures,
    vert: DensityTextures
  ): GPUBindGroup {
    const projectionBindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.computePipelines.projectStageOneData.uniforms,
        },
        {
          binding: 1,
          resource: div.readTexture.createView(),
        },
        {
          binding: 2,
          resource: div.writeTexture.createView(),
        },
        {
          binding: 3,
          resource: pres.prevTexture.createView(),
        },
        {
          binding: 4,
          resource: pres.readTexture.createView(),
        },
        {
          binding: 5,
          resource: pres.writeTexture.createView(),
        },
        {
          binding: 6,
          resource: horiz.readTexture.createView(),
        },
        {
          binding: 7,
          resource: horiz.writeTexture.createView(),
        },
        {
          binding: 8,
          resource: vert.readTexture.createView(),
        },
        {
          binding: 9,
          resource: vert.writeTexture.createView(),
        },
      ],
    });

    return projectionBindGroup;
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
