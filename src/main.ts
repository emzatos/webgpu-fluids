import shaders from "./shaders.wgsl?raw";
import './style.css'


async function main() {
  const adapter = await navigator.gpu?.requestAdapter();
  const device = await adapter?.requestDevice();
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

  const module = device.createShaderModule({
    label: "Shaders for triablge",
    code: shaders,
  });

  const pipeline = device.createRenderPipeline({
    label: 'render pipeling',
    layout: 'auto',
    vertex: {
      module
    },
    fragment: {
      module,
      targets: [{format: presentationFormat}]
    }
  });

  const renderPassDescriptor = {
    label: 'render pass diescript',
    colorAttachments: [
      {
        view: {} as GPUTextureView,
        clearValue: [0.3, 0.3, 0.3, 1.0],
        loadOp: 'clear',
        storeOp: 'store',
      }
    ]
  } satisfies GPURenderPassDescriptor;


  const canvasToSizeMap = new WeakMap();

  function resizeCanvasToDisplaySize(canvas: HTMLCanvasElement) {
    let {width, height} = canvasToSizeMap.get(canvas) || canvas;
    width = Math.max(1, Math.min(width, device?.limits.maxTextureDimension2D!));
    height = Math.max(1, Math.min(height, device?.limits.maxTextureDimension2D!));

    const needResize = canvas.width !== width || canvas.height !== height;
    if (needResize) {
      canvas.width = width;
      canvas.height = height;
    }

    return needResize;
  }

  function render() {
    resizeCanvasToDisplaySize(canvas);
    renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
    const encoder = device?.createCommandEncoder({label: 'commandendocoer'});

    const pass = encoder?.beginRenderPass(renderPassDescriptor);
    pass?.setPipeline(pipeline);
    pass?.end();

    const commandBuffer = encoder?.finish()!;
    device?.queue.submit([commandBuffer])
  }

  const observer = new ResizeObserver(entries => {
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
