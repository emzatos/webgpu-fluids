import { f32, struct, u32 } from "typegpu/data/index";
import shaders from "./shaders.wgsl?raw";
import computeShaders from "./computeShaders.wgsl?raw";
import './style.css'
import tgpu from 'typegpu'

async function main() {
  const adapter = await navigator.gpu?.requestAdapter();
  const device = await adapter?.requestDevice({
    requiredFeatures: ["float32-filterable"]
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
    code: shaders
  });

  const computeModule = device.createShaderModule({
    label: 'Compute texture shaders',
    code: computeShaders
  });

  const renderPipeline = device.createRenderPipeline({
    label: 'render pipeling',
    layout: 'auto',
    vertex: {
      module: renderModule,
    },
    fragment: {
      module: renderModule,
      targets: [{ format: presentationFormat }]
    }
  });

  const computePipeline = device.createComputePipeline({
    label: 'compute pipeline',
    layout: 'auto',
    compute: {
      module: computeModule
    }
  })

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

  const computePassDescriptor = {
    'label': 'Compute Pass Descriptor'
  } satisfies GPUComputePassDescriptor;

  // TGPU Types
  const Uniforms = struct({
    fun: f32,
    time: f32
  });

  // Texture creation

  const textureSize = 128;
  const textureData = new Float32Array({ length: textureSize * textureSize }).map(() => Math.random());
  const texture = device.createTexture({
    'label': 'storage texture',
    'format': 'r32float',
    'size': [textureSize, textureSize],
    'usage': GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
  });

  device.queue.writeTexture({ texture: texture }, textureData, {
    'bytesPerRow': Float32Array.BYTES_PER_ELEMENT * textureSize,
    'rowsPerImage': textureSize
  }, { width: textureSize, height: textureSize });

  const sampler = device.createSampler({
    'magFilter': 'linear',
    'minFilter': 'linear'
  });

  const uniforms = tgpu.createBuffer(Uniforms, {
    fun: Math.random(),
    time: performance.now()
  }).$device(device)
    .$usage(tgpu.Uniform);

  const computeUniforms = tgpu.createBuffer(Uniforms, {
    fun: Math.PI,
    time: performance.now()
  }).$device(device).$usage(tgpu.Uniform);

  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [{
      binding: 0,
      resource: texture.createView()
    },
    {
      binding: 1,
      resource: sampler
    },
    {
      binding: 2,
      resource: uniforms
    }
    ]
  });

  const computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 2,
        resource: computeUniforms
      },
      {
        binding: 0,
        resource: texture.createView()
      },
      {
        binding: 1,
        resource: sampler
      },
    ]
  })


  const canvasToSizeMap = new WeakMap();

  function resizeCanvasToDisplaySize(canvas: HTMLCanvasElement) {
    let { width, height } = canvasToSizeMap.get(canvas) || canvas;
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
    const encoder = device?.createCommandEncoder({ label: 'commandendocoer' });
    const computePass = encoder?.beginComputePass(computePassDescriptor);
    computePass?.setPipeline(computePipeline);
    computePass?.setBindGroup(0, computeBindGroup);
    tgpu.write(computeUniforms, {
      fun: Math.PI,
      time: performance.now() / 200
    })
    computePass?.dispatchWorkgroups(textureSize / 8, textureSize / 8);
    computePass?.end();

    const pass = encoder?.beginRenderPass(renderPassDescriptor);
    pass?.setPipeline(renderPipeline);
    pass?.setBindGroup(0, renderBindGroup);
    pass?.draw(6);
    pass?.end();

    const commandBuffer = encoder?.finish()!;
    device?.queue.submit([commandBuffer])
    requestAnimationFrame(render);
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
