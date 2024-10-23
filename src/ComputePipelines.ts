import diffuseShader from "./shaders/diffuse.wgsl?raw";
import advectShader from "./shaders/advect.wgsl?raw";
import projectShader from "./shaders/project.wgsl?raw";
import splatShader from "./shaders/splat.wgsl?raw";
import { f32, struct, u32, vec2f } from "typegpu/data/index";
import tgpu, { TgpuBuffer } from "typegpu";

export class ComputePipelines {
  public readonly diffusionData: ComputeDeviceData;
  public readonly advectData: ComputeDeviceData;
  public readonly projectStageOneData: ComputeDeviceData;
  public readonly projectStageTwoData: ComputeDeviceData;
  public readonly projectStageThreeData: ComputeDeviceData;
  public readonly splatData: ComputeDeviceData;
  private readonly textureSize: number;

  constructor(device: GPUDevice, textureSize: number) {
    const [diffusionUniform, advectUniform, projectionUniform, splatUniform] =
      this.computeUniforms(device);

    this.textureSize = textureSize;

    this.diffusionData = {
      pipeline: this.createPipeline(device, diffuseShader, "diffuse"),
      descriptor: {
        label: "diffuse pass",
      },
      uniforms: diffusionUniform,
    };

    this.advectData = {
      pipeline: this.createPipeline(device, advectShader, "advect"),
      descriptor: {
        label: "advect pass",
      },
      uniforms: advectUniform,
    };
    this.advectData.uniforms.write({
      dt: 0.02,
      grid_size: textureSize,
      boundaryValue: 2,
    });
    this.projectStageOneData = {
      pipeline: this.createPipeline(device, projectShader, "update_divergence"),
      descriptor: {
        label: "project divergence pass",
      },
      uniforms: projectionUniform,
    };
    this.projectStageOneData.uniforms.write({
      grid_size: textureSize,
      boundaryValue: 2,
    });
    this.projectStageTwoData = {
      pipeline: this.createPipeline(device, projectShader, "update_pressure"),
      descriptor: {
        label: "project pressure pass",
      },
      uniforms: projectionUniform,
    };
    this.projectStageTwoData.uniforms.write({
      grid_size: textureSize,
      boundaryValue: 2,
    });
    this.projectStageThreeData = {
      pipeline: this.createPipeline(
        device,
        projectShader,
        "apply_to_velocities"
      ),
      descriptor: {
        label: "project velocity pass",
      },
      uniforms: projectionUniform,
    };
    this.projectStageThreeData.uniforms.write({
      grid_size: textureSize,
      boundaryValue: 2,
    });
    this.splatData = {
      pipeline: this.createPipeline(device, splatShader, "splat"),
      descriptor: {
        label: "splat pass",
      },
      uniforms: splatUniform,
    };
  }

  private createPipeline = (
    device: GPUDevice,
    shader: string,
    entryPoint: string
  ): GPUComputePipeline => {
    const shaderModule = device.createShaderModule({
      label: entryPoint,
      code: shader,
    });
    const pipeline = device.createComputePipeline({
      label: entryPoint + " pipeline",
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint: entryPoint,
      },
    });

    return pipeline;
  };

  private computeUniforms = (device: GPUDevice): Array<TgpuBuffer<any>> => {
    const diffusionUniforms = tgpu
      .createBuffer(DiffusionUniforms)
      .$device(device)
      .$usage(tgpu.Uniform);
    const advectUniforms = tgpu
      .createBuffer(AdvectUniforms, {
        dt: 0.02,
        grid_size: this.textureSize,
        boundaryValue: 2,
      })
      .$device(device)
      .$usage(tgpu.Uniform);
    const projectionUniforms = tgpu
      .createBuffer(ProjectionUniforms, {
        grid_size: this.textureSize,
        boundaryValue: 2,
      })
      .$device(device)
      .$usage(tgpu.Uniform);
    const splatUniforms = tgpu
      .createBuffer(SplatUniforms)
      .$device(device)
      .$usage(tgpu.Uniform);

    return [
      diffusionUniforms,
      advectUniforms,
      projectionUniforms,
      splatUniforms,
    ];
  };
}

export type ComputeDeviceData = {
  pipeline: GPUComputePipeline;
  descriptor: GPUComputePassDescriptor;
  uniforms: TgpuBuffer<
    | typeof DiffusionUniforms
    | typeof SplatUniforms
    | typeof AdvectUniforms
    | typeof ProjectionUniforms
  >;
};

export const DiffusionUniforms = struct({
  diffusion_constant: f32,
  dt: f32,
  grid_size: u32,
  boundaryValue: u32,
  h: f32,
});

export const SplatUniforms = struct({
  pos: vec2f,
  delta: vec2f,
  radius: f32,
  intensity: f32,
  grid_size: u32,
});

export const AdvectUniforms = struct({
  dt: f32,
  grid_size: u32,
  boundaryValue: u32,
});

export const ProjectionUniforms = struct({
  grid_size: u32,
  boundaryValue: u32,
});
