import { f32, struct, u32 } from "typegpu/data/index";
import shaders from "./shaders.wgsl?raw";
import computeFluidMechanics from "./computeFluidMechanics.wgsl?raw";
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

  // TGPU Types
  const DiffusionUniforms = struct({
    diffusion_constant: f32,
    dt: f32,
    grid_size: u32,
    boundaryValue: u32
  });

  // Texture creation

  const textureSize = 256;
  var densities = new DensityTextures({ device: device, textureSize: textureSize + 2 });
  var velx = new DensityTextures({ device: device, textureSize: textureSize + 2 });

  var temp = new Float32Array({ length: (textureSize + 2) * (textureSize + 2) }).fill(1.0);
  var vely = new DensityTextures({ device: device, textureSize: textureSize + 2, data: temp });

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  const diffusionUniforms = tgpu
    .createBuffer(DiffusionUniforms, {
      diffusion_constant: 0.01,
      dt: 0.003,
      grid_size: textureSize,
      boundaryValue: 0
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

  function performAdvection(encoder: GPUCommandEncoder, dens: DensityTextures, horiz: DensityTextures, vert: DensityTextures) {
    for (var i = 0; i < 1; i++) {
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
            resource: dens.readTexture.createView(),
          },
          {
            binding: 2,
            resource: dens.writeTexture.createView(),
          },
          {
            binding: 3,
            resource: dens.prevTexture.createView(),
          },
          {
            binding: 4,
            resource: horiz.readTexture.createView(),
          },
          {
            binding: 5,
            resource: vert.readTexture.createView(),
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
      dens.swap();
    }

    dens.commit();
  }

  function densityUpdate(encoder: GPUCommandEncoder) {
    performDiffusion(encoder, densities);
    performAdvection(encoder, densities, velx, vely);
  }

  function velocityUpdate(encoder: GPUCommandEncoder) {
    tgpu.write(diffusionUniforms, {
      diffusion_constant: 0.01,
      dt: 0.003,
      grid_size: textureSize,
      boundaryValue: 1
    });

    performDiffusion(encoder, velx);

    tgpu.write(diffusionUniforms, {
      diffusion_constant: 0.01,
      dt: 0.003,
      grid_size: textureSize,
      boundaryValue: 2
    });

    performDiffusion(encoder, vely);

  }

  function render() {
    resizeCanvasToDisplaySize(canvas);
    renderPassDescriptor.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();
    const encoder = device?.createCommandEncoder({ label: "commandendocoer" });

    densityUpdate(encoder!);
    velocityUpdate(encoder!);

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
