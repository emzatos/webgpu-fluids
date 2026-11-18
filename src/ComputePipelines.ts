import diffuseShader from "./shaders/diffuse.wgsl?raw";
import advectShader from "./shaders/advect.wgsl?raw";
import projectShader from "./shaders/project.wgsl?raw";
import splatShader from "./shaders/splat.wgsl?raw";
import { d, type TgpuRoot } from "typegpu";

export class ComputePipelines {
  readonly diffusionData;
  readonly advectData;
  readonly projectStageOneData;
  readonly projectStageTwoData;
  readonly projectStageThreeData;
  readonly splatData;

  constructor(root: TgpuRoot, textureSize: number) {
    const { device } = root;
    const createModule = (label: string, code: string) =>
      device.createShaderModule({ label, code });

    const createPipeline = (module: GPUShaderModule, entryPoint: string) =>
      device.createComputePipeline({
        label: entryPoint,
        layout: "auto",
        compute: { module, entryPoint },
      });

    const project = createModule("projection", projectShader);
    const projectionUniforms = root.createBuffer(ProjectionUniforms, {
      grid_size: textureSize,
      boundaryValue: 2,
    }).$usage("uniform");
    const projectionData = (entryPoint: string) => ({
      pipeline: createPipeline(project, entryPoint),
      descriptor: { label: entryPoint },
      uniforms: projectionUniforms,
    });

    this.diffusionData = {
      pipeline: createPipeline(createModule("diffusion", diffuseShader), "diffuse"),
      descriptor: { label: "diffuse pass" },
      uniforms: root.createBuffer(DiffusionUniforms).$usage("uniform"),
    };
    this.advectData = {
      pipeline: createPipeline(createModule("advection", advectShader), "advect"),
      descriptor: { label: "advect pass" },
      uniforms: root.createBuffer(AdvectUniforms, {
        dt: 0.02,
        grid_size: textureSize,
        boundaryValue: 2,
      }).$usage("uniform"),
    };
    this.projectStageOneData = projectionData("update_divergence");
    this.projectStageTwoData = projectionData("update_pressure");
    this.projectStageThreeData = projectionData("apply_to_velocities");
    this.splatData = {
      pipeline: createPipeline(createModule("splat", splatShader), "splat"),
      descriptor: { label: "splat pass" },
      uniforms: root.createBuffer(SplatUniforms).$usage("uniform"),
    };
  }
}

export const DiffusionUniforms = d.struct({
  diffusion_constant: d.f32,
  dt: d.f32,
  grid_size: d.u32,
  boundaryValue: d.u32,
  h: d.f32,
});

export const SplatUniforms = d.struct({
  pos: d.vec2f,
  delta: d.vec2f,
  radius: d.f32,
  intensity: d.f32,
  grid_size: d.u32,
});

export const AdvectUniforms = d.struct({
  dt: d.f32,
  grid_size: d.u32,
  boundaryValue: d.u32,
});

export const ProjectionUniforms = d.struct({
  grid_size: d.u32,
  boundaryValue: d.u32,
});
