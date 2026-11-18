import diffuseShader from "./shaders/diffuse.wgsl?raw";
import advectShader from "./shaders/advect.wgsl?raw";
import projectShader from "./shaders/project.wgsl?raw";
import splatShader from "./shaders/splat.wgsl?raw";
import { d, tgpu, type TgpuBindGroupLayout, type TgpuRoot } from "typegpu";

export class ComputePipelines {
  readonly diffusionData;
  readonly velocityDiffusionUniforms;
  readonly densityDiffusionUniforms;
  readonly advectData;
  readonly divergencePipeline;
  readonly pressurePipeline;
  readonly projectionPipeline;
  readonly projectionUniforms;
  readonly splatData;

  constructor(root: TgpuRoot, textureSize: number) {
    const { device } = root;
    const createModule = (label: string, code: string) =>
      device.createShaderModule({ label, code });

    const createPipeline = (
      module: GPUShaderModule,
      entryPoint: string,
      layout?: TgpuBindGroupLayout
    ) =>
      device.createComputePipeline({
        label: entryPoint,
        layout: layout ? device.createPipelineLayout({ bindGroupLayouts: [root.unwrap(layout)] }) : "auto",
        compute: { module, entryPoint },
      });

    const project = createModule("projection", projectShader);
    this.projectionUniforms = root.createBuffer(ProjectionUniforms, {
      grid_size: textureSize,
    }).$usage("uniform");

    this.diffusionData = {
      pipeline: createPipeline(createModule("diffusion", diffuseShader), "diffuse"),
      descriptor: { label: "diffuse pass" },
    };
    const diffusionUniforms = (dt: number) => root.createBuffer(DiffusionUniforms, {
      diffusion_constant: 0.0001,
      dt,
      grid_size: textureSize,
      boundaryValue: 2,
    }).$usage("uniform");
    this.velocityDiffusionUniforms = diffusionUniforms(0.002);
    this.densityDiffusionUniforms = diffusionUniforms(0.02);
    this.advectData = {
      pipeline: createPipeline(createModule("advection", advectShader), "advect"),
      descriptor: { label: "advect pass" },
      uniforms: root.createBuffer(AdvectUniforms, {
        dt: 0.02,
        grid_size: textureSize,
        boundaryValue: 2,
      }).$usage("uniform"),
    };
    this.divergencePipeline = createPipeline(project, "update_divergence", divergenceLayout);
    this.pressurePipeline = createPipeline(project, "update_pressure", pressureLayout);
    this.projectionPipeline = createPipeline(project, "apply_to_velocities", projectionLayout);
    this.splatData = {
      pipeline: createPipeline(createModule("splat", splatShader), "splat"),
      descriptor: { label: "splat pass" },
      uniforms: root.createBuffer(SplatUniforms).$usage("uniform"),
    };
  }
}

const DiffusionUniforms = d.struct({
  diffusion_constant: d.f32,
  dt: d.f32,
  grid_size: d.u32,
  boundaryValue: d.u32,
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

const ProjectionUniforms = d.struct({ grid_size: d.u32 });
const projectionUniform = { uniform: ProjectionUniforms };
const readField = { texture: d.texture2d(d.f32), sampleType: "unfilterable-float" as const };
const writeField = { storageTexture: d.textureStorage2d("r32float") };

// Keep the binding indices in sync with project.wgsl. Null entries reserve unused slots.
export const divergenceLayout = tgpu.bindGroupLayout({
  uniforms: projectionUniform,
  divergence: null,
  divergenceOutput: writeField,
  previousPressure: readField,
  pressure: null,
  pressureOutput: writeField,
  horizontal: readField,
  horizontalOutput: null,
  vertical: readField,
});
export const pressureLayout = tgpu.bindGroupLayout({
  uniforms: projectionUniform,
  divergence: readField,
  divergenceOutput: null,
  previousPressure: null,
  pressure: readField,
  pressureOutput: writeField,
});
export const projectionLayout = tgpu.bindGroupLayout({
  uniforms: projectionUniform,
  divergence: null,
  divergenceOutput: null,
  previousPressure: null,
  pressure: readField,
  pressureOutput: null,
  horizontal: readField,
  horizontalOutput: writeField,
  vertical: readField,
  verticalOutput: writeField,
});
