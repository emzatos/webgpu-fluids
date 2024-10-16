import { f32, struct, u32 } from "typegpu/data/index";
import shaders from "./shaders.wgsl?raw";
import computeFluidMechanics from "./computeFluidMechanics.wgsl?raw";
import computeProject from "./computeProject.wgsl?raw";
import "./style.css";
import tgpu from "typegpu";
import { DensityTextures } from "./DensityTextures";


async function main() {
  const adapter = await navigator.gpu?.requestAdapter();
  const device = await adapter?.requestDevice({
    requiredFeatures: ["float32-filterable"],
  });
  if (!device) {
    alert("WebGPU not supported on this device");
    return;
  }

  const canvas = document.querySelector("canvas") as HTMLCanvasElement;
  const context = canvas.getContext("webgpu") as GPUCanvasContext;
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  context?.configure({
    device,
    format: presentationFormat,
  });

  // shaders and pipeline
  const renderModule = device.createShaderModule({
    label: "Shaders for triablge",
    code: shaders,
  });

  const computeFluidModule = device.createShaderModule({
    label: "fluid compute module",
    code: computeFluidMechanics,
  });

  const computeProjectModule = device.createShaderModule({
    label: 'compute projection',
    code: computeProject
  });

  const renderPipeline = device.createRenderPipeline({
    label: "render pipeling",
    layout: "auto",
    vertex: {
      module: renderModule,
    },
    fragment: {
      module: renderModule,
      targets: [{ format: presentationFormat }],
    },
  });

  const diffusionPipeline = device.createComputePipeline({
    label: "diffusion pipeline",
    layout: "auto",
    compute: {
      module: computeFluidModule,
      entryPoint: "diffuse",
    },
  });

  const advectPipeline = device.createComputePipeline({
    label: "advect pipeline",
    layout: "auto",
    compute: {
      module: computeFluidModule,
      entryPoint: "advect",
    },
  });

  const projectPipelineBegin = device.createComputePipeline({
    label: "project pipeline beginning",
    layout: "auto",
    compute: {
      module: computeProjectModule,
      entryPoint: "project_begin",
    },
  });

  const projectPipelineMiddle = device.createComputePipeline({
    label: "project pipeline middlie",
    layout: "auto",
    compute: {
      module: computeProjectModule,
      entryPoint: "project_middle",
    },
  });

  const projectPipelineEnd = device.createComputePipeline({
    label: "project pipeline end",
    layout: "auto",
    compute: {
      module: computeProjectModule,
      entryPoint: "project_end",
    },
  });

  const renderPassDescriptor = {
    label: "render pass diescript",
    colorAttachments: [
      {
        view: {} as GPUTextureView,
        clearValue: [0.3, 0.3, 0.3, 1.0],
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  } satisfies GPURenderPassDescriptor;

  const diffusePassDescriptor = {
    label: "Diffusion Pass Descriptor",
  } satisfies GPUComputePassDescriptor;

  const advectPassDescriptor = {
    label: "advect Pass Descriptor",
  } satisfies GPUComputePassDescriptor;

  const projectBeginPassDescriptor = {
    label: "project begin Pass Descriptor",
  } satisfies GPUComputePassDescriptor;

  const projectMiddlePassDescriptor = {
    label: "project middle Pass Descriptor",
  } satisfies GPUComputePassDescriptor;

  const projectEndPassDescriptor = {
    label: "project end Pass Descriptor",
  } satisfies GPUComputePassDescriptor;

  // TGPU Types
  const DiffusionUniforms = struct({
    diffusion_constant: f32,
    dt: f32,
    grid_size: u32,
    boundaryValue: u32,
    h: f32
  });

  // Texture creation

  const textureSize = 256;
  var densities = new DensityTextures({ device: device, textureSize: textureSize + 2 });
  var velx = new DensityTextures({ device: device, textureSize: textureSize + 2 });

  var temp = new Float32Array({ length: (textureSize + 2) * (textureSize + 2) }).fill(1.0);
  var vely = new DensityTextures({ device: device, textureSize: textureSize + 2 });

  temp = new Float32Array({ length: (textureSize + 2) * (textureSize + 2) }).fill(0.0);

  var divergence = new DensityTextures({ device: device, textureSize: textureSize + 2, data: temp });
  var pressure = new DensityTextures({ device: device, textureSize: textureSize + 2, data: temp });

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  const diffusionUniforms = tgpu
    .createBuffer(DiffusionUniforms, {
      diffusion_constant: 0.01,
      dt: 0.003,
      grid_size: textureSize,
      boundaryValue: 0,
      h: 1.0 / textureSize
    })
    .$device(device)
    .$usage(tgpu.Uniform);

  const canvasToSizeMap = new WeakMap();

  function resizeCanvasToDisplaySize(canvas: HTMLCanvasElement) {
    let { width, height } = canvasToSizeMap.get(canvas) || canvas;
    width = Math.max(1, Math.min(width, device?.limits.maxTextureDimension2D!));
    height = Math.max(
      1,
      Math.min(height, device?.limits.maxTextureDimension2D!),
    );

    const needResize = canvas.width !== width || canvas.height !== height;
    if (needResize) {
      canvas.width = width;
      canvas.height = height;
    }

    return needResize;
  }

  function performDiffusion(encoder: GPUCommandEncoder, dens: DensityTextures) {
    for (var i = 0; i < 10; i++) {
      const diffusionPass = encoder.beginComputePass(diffusePassDescriptor);

      const diffusionBindGroup = device!.createBindGroup({
        layout: diffusionPipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: diffusionUniforms
          },
          {
            binding: 1,
            resource: dens.readTexture.createView()
          },
          {
            binding: 2,
            resource: dens.writeTexture.createView()
          },
          {
            binding: 3,
            resource: dens.prevTexture.createView()
          },
        ],
      });

      diffusionPass.setPipeline(diffusionPipeline);
      diffusionPass.setBindGroup(0, diffusionBindGroup);
      diffusionPass.dispatchWorkgroups(textureSize / 8, textureSize / 8);
      diffusionPass.end();
      dens.swap();
    }

    dens.commit();
  }

  function performAdvection(encoder: GPUCommandEncoder, prev_grid: GPUTexture, new_grid: GPUTexture, horizontal: GPUTexture, vertical: GPUTexture) {
    const advectPass = encoder.beginComputePass(advectPassDescriptor);

    const advectBindGroup = device!.createBindGroup({
      layout: advectPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: diffusionUniforms,
        },
        {
          binding: 1,
          resource: prev_grid.createView(),
        },
        {
          binding: 2,
          resource: new_grid.createView(),
        },
        {
          binding: 3,
          resource: prev_grid.createView(),
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
          resource: sampler,
        },
      ],
    });

    advectPass.setPipeline(advectPipeline);
    advectPass.setBindGroup(0, advectBindGroup);
    advectPass.dispatchWorkgroups(textureSize / 8, textureSize / 8);
    advectPass.end();
  }

  function createProjectionBindGroup(pipeline: GPUComputePipeline, div: DensityTextures, pres: DensityTextures, horiz: DensityTextures, vert: DensityTextures): GPUBindGroup {
    const projectionBindGroup = device!.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: diffusionUniforms,
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
          resource: pres.readTexture.createView(),
        },
        {
          binding: 4,
          resource: pres.writeTexture.createView(),
        },
        {
          binding: 5,
          resource: horiz.readTexture.createView(),
        },
        {
          binding: 6,
          resource: horiz.writeTexture.createView(),
        },
        {
          binding: 7,
          resource: vert.readTexture.createView(),
        },
        {
          binding: 8,
          resource: vert.writeTexture.createView(),
        },
      ],
    });

    return projectionBindGroup;
  }


  function performBeginProjection(encoder: GPUCommandEncoder, div: DensityTextures, pres: DensityTextures, horiz: DensityTextures, vert: DensityTextures) {
    const projectionPass = encoder.beginComputePass(projectBeginPassDescriptor);

    const projectionBindGroup = createProjectionBindGroup(projectPipelineBegin, div, pres, horiz, vert);

    projectionPass.setPipeline(projectPipelineBegin);
    projectionPass.setBindGroup(0, projectionBindGroup);
    projectionPass.dispatchWorkgroups(textureSize / 8, textureSize / 8);
    projectionPass.end();
    pres.swap();
    div.swap();
  }

  function performMiddleProjection(encoder: GPUCommandEncoder, div: DensityTextures, pres: DensityTextures, horiz: DensityTextures, vert: DensityTextures) {

    for (var i = 0; i < 20; i++) {
      const projectionPass = encoder.beginComputePass(projectMiddlePassDescriptor);

      const projectionBindGroup = createProjectionBindGroup(projectPipelineMiddle, div, pres, horiz, vert);

      projectionPass.setPipeline(projectPipelineMiddle);
      projectionPass.setBindGroup(0, projectionBindGroup);
      projectionPass.dispatchWorkgroups(textureSize / 8, textureSize / 8);
      projectionPass.end();
      pres.swap();
    }
  }

  function performEndProjection(encoder: GPUCommandEncoder, div: DensityTextures, pres: DensityTextures, horiz: DensityTextures, vert: DensityTextures) {
    const projectionPass = encoder.beginComputePass(projectEndPassDescriptor);

    const projectionBindGroup = createProjectionBindGroup(projectPipelineEnd, div, pres, horiz, vert);

    projectionPass.setPipeline(projectPipelineEnd);
    projectionPass.setBindGroup(0, projectionBindGroup);
    projectionPass.dispatchWorkgroups(textureSize / 8, textureSize / 8);
    projectionPass.end();
    horiz.swap();
    vert.swap();
  }

  function performProjection(encoder: GPUCommandEncoder) {
    performBeginProjection(encoder, divergence, pressure, velx, vely);
    performMiddleProjection(encoder, divergence, pressure, velx, vely);
    performEndProjection(encoder, divergence, pressure, velx, vely);
  }

  function densityUpdate(encoder: GPUCommandEncoder) {
    //   performDiffusion(encoder, densities);
    densities.commit();
    performAdvection(encoder, densities.readTexture, densities.writeTexture, velx.readTexture, vely.readTexture);
    densities.swap();
    // densities.swap();
    // densities.commit();
  }

  function velocityUpdate(encoder: GPUCommandEncoder) {
    // tgpu.write(diffusionUniforms, {
    //   diffusion_constant: 0.01,
    //   dt: 0.003,
    //   grid_size: textureSize,
    //   boundaryValue: 1,
    //   h: 1.0 / textureSize
    // });

    // performDiffusion(encoder, velx);

    // tgpu.write(diffusionUniforms, {
    //   diffusion_constant: 0.01,
    //   dt: 0.003,
    //   grid_size: textureSize,
    //   boundaryValue: 2,
    //   h: 1.0 / textureSize
    // });

    // performDiffusion(encoder, vely);

    performProjection(encoder);

    velx.commit()
    performAdvection(encoder, velx.readTexture, velx.writeTexture, velx.prevTexture, vely.prevTexture);
    velx.swap();
    vely.commit();
    performAdvection(encoder, vely.readTexture, vely.writeTexture, velx.prevTexture, vely.prevTexture);
    vely.swap();
    //   performProjection(encoder)

  }

  function render() {
    resizeCanvasToDisplaySize(canvas);
    renderPassDescriptor.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();
    const encoder = device?.createCommandEncoder({ label: "commandendocoer" });

    velocityUpdate(encoder!);
    densityUpdate(encoder!);

    const renderBindGroup = device!.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: densities.readTexture.createView(),
        },
        {
          binding: 1,
          resource: sampler,
        },
      ],
    });

    const pass = encoder?.beginRenderPass(renderPassDescriptor);
    pass?.setPipeline(renderPipeline);
    pass?.setBindGroup(0, renderBindGroup);
    pass?.draw(6);
    pass?.end();

    const commandBuffer = encoder?.finish()!;
    device?.queue.submit([commandBuffer]);
    requestAnimationFrame(render);
  }

  const observer = new ResizeObserver((entries) => {
    for (const entry of entries) {
      canvasToSizeMap.set(entry.target, {
        width: entry.contentBoxSize[0].inlineSize,
        height: entry.contentBoxSize[0].blockSize,
      });
    }
    render();
  });

  observer.observe(canvas);
}

main();
