import { f32, struct } from "typegpu/data/index";
import shaders from "./shaders.wgsl?raw";
import "./style.css";
import tgpu from "typegpu";
import { MouseHandler } from "./MouseHandler";
import { ComputePipelines } from "./ComputePipelines";
import { FluidSimulation } from "./FluidSimulation";

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

  const simulation = new FluidSimulation(device, textureSize);
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

  function render() {
    resizeCanvasToDisplaySize(canvas);
    renderPassDescriptor.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();
    const encoder = device?.createCommandEncoder({ label: "commandendocoer" });

    // call splat with mouse position
    if (mouseHandler.mouseDown) {
      simulation.splat(
        encoder!,
        mouseHandler.mousePosition.x,
        mouseHandler.mousePosition.y,
        (mouseHandler.mousePosition.x - mouseHandler.lastMousePosition.x) * 0.1,
        (mouseHandler.mousePosition.y - mouseHandler.lastMousePosition.y) * 0.1,
        textureSize / 50,
        1
      );
    }

    simulation.update(encoder!);
    fragmentUniforms.write({ time: performance.now() / 2000 });

    const renderBindGroup = device!.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: simulation.getDensityTexture().createView(),
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
