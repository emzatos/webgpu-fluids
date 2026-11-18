import boundaryShader from "./shaders/boundary.wgsl?raw";
import diffuseShader from "./shaders/diffuse.wgsl?raw";
import advectShader from "./shaders/advect.wgsl?raw";
import projectShader from "./shaders/project.wgsl?raw";
import splatShader from "./shaders/splat.wgsl?raw";
import { d, tgpu, type TgpuBindGroupLayout, type TgpuRoot } from "typegpu";

export class ComputePipelines {
  readonly diffusionData;
  readonly fieldUniforms;
  readonly advectData;
  readonly divergencePipeline;
  readonly pressurePipeline;
  readonly projectionPipeline;
  readonly projectionUniforms;
  readonly splatData;

  constructor(root: TgpuRoot, textureSize: number) {
    const { device } = root;
    const createModule = (label: string, code: string) =>
      device.createShaderModule({ label, code: boundaryShader + "\n" + code });

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
    const fieldUniforms = (boundaryValue: number, diffusionDt: number) => ({
      diffusion: root.createBuffer(DiffusionUniforms, {
        diffusion_constant: 0.0001, dt: diffusionDt, grid_size: textureSize, boundaryValue,
      }).$usage("uniform"),
      advection: root.createBuffer(AdvectUniforms, {
        dt: 0.02, grid_size: textureSize, boundaryValue,
      }).$usage("uniform"),
    });
    this.fieldUniforms = {
      density: fieldUniforms(0, 0.02),
      horizontal: fieldUniforms(1, 0.002),
      vertical: fieldUniforms(2, 0.002),
    };
    this.advectData = {
      pipeline: createPipeline(createModule("advection", advectShader), "advect"),
      descriptor: { label: "advect pass" },
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
