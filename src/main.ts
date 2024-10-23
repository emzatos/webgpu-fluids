import { f32, struct, vec2f } from "typegpu/data/index";
import shaders from "./shaders.wgsl?raw";
import "./style.css";
import tgpu from "typegpu";
import { DensityTextures } from "./DensityTextures";
import { MouseHandler } from "./MouseHandler";
import { ComputePipelines } from "./ComputePipelines";

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
  const textureSize = 1024;

  const computePipelines = new ComputePipelines(device, 1024);
  const mouseHandler = new MouseHandler(canvas, textureSize);
  // shaders and pipeline
  const renderModule = device.createShaderModule({
    label: "Shaders for triablge",
    code: shaders,
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

  // TGPU Types

  const FragmentUniforms = struct({
    time: f32,
  });

  function initializeVelocity(
    textureSize: number
  ): [Float32Array, Float32Array] {
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
  }

  // Texture creation

  const velocityValues = initializeVelocity(textureSize);
  var densities = new DensityTextures({
    device: device,
    textureSize: textureSize + 2,
  });
  var velx = new DensityTextures({
    device: device,
    textureSize: textureSize + 2,
    data: velocityValues[0],
  });

  var vely = new DensityTextures({
    device: device,
    textureSize: textureSize + 2,
    data: velocityValues[1],
  });

  var temp = new Float32Array({
    length: (textureSize + 2) * (textureSize + 2),
  }).fill(0.0);

  var divergence = new DensityTextures({
    device: device,
    textureSize: textureSize + 2,
    data: temp,
  });
  var pressure = new DensityTextures({
    device: device,
    textureSize: textureSize + 2,
    data: temp,
  });

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  const fragmentUniforms = tgpu
    .createBuffer(FragmentUniforms)
    .$device(device)
    .$usage(tgpu.Uniform);

  const canvasToSizeMap = new WeakMap();

  function resizeCanvasToDisplaySize(canvas: HTMLCanvasElement) {
    let { width, height } = canvasToSizeMap.get(canvas) || canvas;
    width = Math.max(1, Math.min(width, device?.limits.maxTextureDimension2D!));
    height = Math.max(
      1,
      Math.min(height, device?.limits.maxTextureDimension2D!)
    );

    const needResize = canvas.width !== width || canvas.height !== height;
    if (needResize) {
      canvas.width = width;
      canvas.height = height;
    }

    return needResize;
  }

  function performDiffusion(
    encoder: GPUCommandEncoder,
    dens: DensityTextures,
    diffusion: number,
    dt: number
  ) {
    computePipelines.diffusionData.uniforms.write({
      diffusion_constant: diffusion,
      dt: dt,
      grid_size: textureSize,
      boundaryValue: 2,
      h: 1.0 / textureSize,
    });

    for (var i = 0; i < 20; i++) {
      const diffusionPass = encoder.beginComputePass(
        computePipelines.diffusionData.descriptor
      );

      const diffusionBindGroup = device!.createBindGroup({
        layout: computePipelines.diffusionData.pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: computePipelines.diffusionData.uniforms,
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
        ],
      });

      diffusionPass.setPipeline(computePipelines.diffusionData.pipeline);
      diffusionPass.setBindGroup(0, diffusionBindGroup);
      diffusionPass.dispatchWorkgroups(textureSize / 8, textureSize / 8);
      diffusionPass.end();
      dens.flip();
    }
  }

  function performAdvection(
    encoder: GPUCommandEncoder,
    grid: DensityTextures,
    horizontal: GPUTexture,
    vertical: GPUTexture
  ) {
    const advectPass = encoder.beginComputePass(
      computePipelines.advectData.descriptor
    );

    const advectBindGroup = device!.createBindGroup({
      layout: computePipelines.advectData.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: computePipelines.advectData.uniforms,
        },
        {
          binding: 1,
          resource: grid.readTexture.createView(),
        },
        {
          binding: 2,
          resource: grid.writeTexture.createView(),
        },
        {
          binding: 3,
          resource: grid.prevTexture.createView(), //unused
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

    advectPass.setPipeline(computePipelines.advectData.pipeline);
    advectPass.setBindGroup(0, advectBindGroup);
    advectPass.dispatchWorkgroups(textureSize / 8, textureSize / 8);
    advectPass.end();
    grid.flip();
  }

  function createProjectionBindGroup(
    pipeline: GPUComputePipeline,
    div: DensityTextures,
    pres: DensityTextures,
    horiz: DensityTextures,
    vert: DensityTextures
  ): GPUBindGroup {
    const projectionBindGroup = device!.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: computePipelines.projectStageOneData.uniforms,
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

  function performBeginProjection(
    encoder: GPUCommandEncoder,
    div: DensityTextures,
    pres: DensityTextures,
    horiz: DensityTextures,
    vert: DensityTextures
  ) {
    const projectionPass = encoder.beginComputePass(
      computePipelines.projectStageOneData.descriptor
    );

    const projectionBindGroup = createProjectionBindGroup(
      computePipelines.projectStageOneData.pipeline,
      div,
      pres,
      horiz,
      vert
    );

    projectionPass.setPipeline(computePipelines.projectStageOneData.pipeline);
    projectionPass.setBindGroup(0, projectionBindGroup);
    projectionPass.dispatchWorkgroups(textureSize / 8, textureSize / 8);
    projectionPass.end();
    pres.flip();
    pres.swap();
    div.flip();
  }

  function performMiddleProjection(
    encoder: GPUCommandEncoder,
    div: DensityTextures,
    pres: DensityTextures,
    horiz: DensityTextures,
    vert: DensityTextures
  ) {
    for (var i = 0; i < 200; i++) {
      const projectionPass = encoder.beginComputePass(
        computePipelines.projectStageTwoData.descriptor
      );

      const projectionBindGroup = createProjectionBindGroup(
        computePipelines.projectStageTwoData.pipeline,
        div,
        pres,
        horiz,
        vert
      );

      projectionPass.setPipeline(computePipelines.projectStageTwoData.pipeline);
      projectionPass.setBindGroup(0, projectionBindGroup);
      projectionPass.dispatchWorkgroups(textureSize / 8, textureSize / 8);
      projectionPass.end();
      pres.flip();
    }
  }

  function performEndProjection(
    encoder: GPUCommandEncoder,
    div: DensityTextures,
    pres: DensityTextures,
    horiz: DensityTextures,
    vert: DensityTextures
  ) {
    const projectionPass = encoder.beginComputePass(
      computePipelines.projectStageThreeData.descriptor
    );

    const projectionBindGroup = createProjectionBindGroup(
      computePipelines.projectStageThreeData.pipeline,
      div,
      pres,
      horiz,
      vert
    );

    projectionPass.setPipeline(computePipelines.projectStageThreeData.pipeline);
    projectionPass.setBindGroup(0, projectionBindGroup);
    projectionPass.dispatchWorkgroups(textureSize / 8, textureSize / 8);
    projectionPass.end();
    horiz.flip();
    vert.flip();
  }

  function performProjection(encoder: GPUCommandEncoder) {
    performBeginProjection(encoder, divergence, pressure, velx, vely);
    performMiddleProjection(encoder, divergence, pressure, velx, vely);
    performEndProjection(encoder, divergence, pressure, velx, vely);
  }

  function splat(
    encoder: GPUCommandEncoder,
    x: number,
    y: number,
    dx: number,
    dy: number,
    radius: number,
    intensity: number
  ) {
    const splatPass = encoder.beginComputePass(
      computePipelines.splatData.descriptor
    );

    computePipelines.splatData.uniforms.write({
      pos: vec2f(x, y),
      delta: vec2f(dx, dy),
      radius: radius,
      intensity: intensity,
      grid_size: textureSize,
    });

    const splatBindGroup = device!.createBindGroup({
      layout: computePipelines.splatData.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: computePipelines.splatData.uniforms,
        },
        {
          binding: 1,
          resource: densities.readTexture.createView(),
        },
        {
          binding: 2,
          resource: densities.writeTexture.createView(),
        },
        {
          binding: 3,
          resource: velx.readTexture.createView(),
        },
        {
          binding: 4,
          resource: velx.writeTexture.createView(),
        },
        {
          binding: 5,
          resource: vely.readTexture.createView(),
        },
        {
          binding: 6,
          resource: vely.writeTexture.createView(),
        },
      ],
    });

    splatPass.setPipeline(computePipelines.splatData.pipeline);
    splatPass.setBindGroup(0, splatBindGroup);
    splatPass.dispatchWorkgroups(textureSize / 8, textureSize / 8);
    splatPass.end();

    encoder.copyTextureToTexture(
      {
        texture: densities.writeTexture,
      },
      {
        texture: densities.prevTexture,
      },
      {
        width: textureSize,
        height: textureSize,
      }
    );
    encoder.copyTextureToTexture(
      {
        texture: velx.writeTexture,
      },
      {
        texture: velx.prevTexture,
      },
      {
        width: textureSize,
        height: textureSize,
      }
    );
    encoder.copyTextureToTexture(
      {
        texture: vely.writeTexture,
      },
      {
        texture: vely.prevTexture,
      },
      {
        width: textureSize,
        height: textureSize,
      }
    );

    densities.flip();
    velx.flip();
    vely.flip();
  }

  function densityUpdate(encoder: GPUCommandEncoder) {
    densities.swap();
    performDiffusion(encoder, densities, 0.0001, 0.02);
    densities.swap();
    performAdvection(encoder, densities, velx.readTexture, vely.readTexture);
    densities.flip();
  }

  function velocityUpdate(encoder: GPUCommandEncoder) {
    velx.swap();
    vely.swap();

    performDiffusion(encoder, velx, 0.0001, 0.02);

    performDiffusion(encoder, vely, 0.0001, 0.02);
    performProjection(encoder);

    velx.swap();
    vely.swap();

    performAdvection(encoder, velx, velx.prevTexture, vely.prevTexture);
    performAdvection(encoder, vely, velx.prevTexture, vely.prevTexture);
    performProjection(encoder);
  }

  function render() {
    resizeCanvasToDisplaySize(canvas);
    renderPassDescriptor.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();
    const encoder = device?.createCommandEncoder({ label: "commandendocoer" });

    // call splat with mouse position
    if (mouseHandler.mouseDown) {
      splat(
        encoder!,
        mouseHandler.mousePosition.x,
        mouseHandler.mousePosition.y,
        (mouseHandler.mousePosition.x - mouseHandler.lastMousePosition.x) * 0.1,
        (mouseHandler.mousePosition.y - mouseHandler.lastMousePosition.y) * 0.1,
        textureSize / 50,
        1
      );
      densities.swap();
      velx.swap();
      vely.swap();
    }

    velocityUpdate(encoder!);
    densityUpdate(encoder!);
    fragmentUniforms.write({ time: performance.now() / 2000 });

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
        {
          binding: 2,
          resource: fragmentUniforms,
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
    mouseHandler.lastMousePosition = {
      x: mouseHandler.mousePosition.x,
      y: mouseHandler.mousePosition.y,
    };
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
