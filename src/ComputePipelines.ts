import boundaryShader from "./shaders/boundary.wgsl?raw";
import diffuseShader from "./shaders/diffuse.wgsl?raw";
import advectShader from "./shaders/advect.wgsl?raw";
import projectShader from "./shaders/project.wgsl?raw";
import splatShader from "./shaders/splat.wgsl?raw";
import { MAX_STEPS_PER_FRAME } from "./SimulationClock";
import { d, tgpu, type TgpuBindGroupLayout, type TgpuRoot } from "typegpu";

export const ADVECTION_DT = 0.02;

export class ComputePipelines {
  readonly diffusionData;
  readonly fieldUniforms;
  readonly advectData;
  readonly divergencePipeline;
  readonly pressurePipeline;
  readonly projectionPipeline;
  readonly projectionUniforms;
  readonly splatData;

  constructor(root: TgpuRoot, textureSize: number, dyeSize: number) {
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
      pipeline: createPipeline(createModule("diffusion", diffuseShader), "diffuse", diffusionLayout),
      descriptor: { label: "diffuse pass" },
    };
    const diffusionUniforms = (gridSize: number, boundaryValue: number, dt: number) =>
      root.createBuffer(DiffusionUniforms, {
        diffusion_constant: 0.0001, dt, grid_size: gridSize, boundaryValue,
      }).$usage("uniform");
    const velocityUniforms = (boundaryValue: number) => ({
      diffusion: diffusionUniforms(textureSize, boundaryValue, 0.002),
      advection: root.createBuffer(AdvectUniforms, {
        dt: ADVECTION_DT, grid_size: textureSize, boundaryValue,
      }).$usage("uniform"),
    });
    this.fieldUniforms = {
      density: diffusionUniforms(dyeSize, 0, 0.02),
      horizontal: velocityUniforms(1),
      vertical: velocityUniforms(2),
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
      // Each step recorded before submission needs its own queue-written values.
      uniforms: Array.from({ length: MAX_STEPS_PER_FRAME }, () =>
        root.createBuffer(SplatUniforms).$usage("uniform")
      ),
    };
  }
  destroy() {
    const { density, horizontal, vertical } = this.fieldUniforms;
    density.destroy();
    for (const { diffusion, advection } of [horizontal, vertical]) {
      diffusion.destroy();
      advection.destroy();
    }
    this.projectionUniforms.destroy();
    for (const uniform of this.splatData.uniforms) uniform.destroy();
  }
}

const DiffusionUniforms = d.struct({
  diffusion_constant: d.f32,
  dt: d.f32,
  grid_size: d.u32,
  boundaryValue: d.u32,
});

export const SplatUniforms = d.struct({
  start: d.vec2f,
  end: d.vec2f,
  radius: d.f32,
  intensity: d.f32,
  grid_size: d.u32,
  dye_size: d.u32,
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

export const diffusionLayout = tgpu.bindGroupLayout({
  uniforms: { uniform: DiffusionUniforms },
  estimate: readField,
  output: writeField,
  source: readField,
});

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
