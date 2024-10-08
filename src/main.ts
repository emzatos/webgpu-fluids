import { f32, struct, u32 } from "typegpu/data/index";
import shaders from "./shaders.wgsl?raw";
import computeShaders from "./computeShaders.wgsl?raw";
import computeFluidMechanics from "./computeFluidMechanics.wgsl?raw";
import './style.css'
import tgpu from 'typegpu'


const createFluidTexture = (device: GPUDevice, textureSize: number): [GPUTexture, GPUTexture, GPUTexture] => {
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

  const texture2 = device.createTexture({
    'label': 'storage texture2',
    'format': 'r32float',
    'size': [textureSize, textureSize],
    'usage': GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
  });

  device.queue.writeTexture({ texture: texture2 }, textureData, {
    'bytesPerRow': Float32Array.BYTES_PER_ELEMENT * textureSize,
    'rowsPerImage': textureSize
  }, { width: textureSize, height: textureSize });

  const texture3 = device.createTexture({
    'label': 'storage texture3',
    'format': 'r32float',
    'size': [textureSize, textureSize],
    'usage': GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
  });

  device.queue.writeTexture({ texture: texture3 }, textureData, {
    'bytesPerRow': Float32Array.BYTES_PER_ELEMENT * textureSize,
    'rowsPerImage': textureSize
  }, { width: textureSize, height: textureSize });

  return [texture, texture2, texture3];
}

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

  const computeFluidModule = device.createShaderModule({
    label: 'fluid compute module',
    code: computeFluidMechanics,
  })

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
  });

  const diffusionPipeline = device.createComputePipeline({
    label: 'diffusion pipeline',
    layout: 'auto',
    compute: {
      module: computeFluidModule,
      entryPoint: 'diffuse'
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

  const diffusePassDescriptor = {
    'label': 'Diffusion Pass Descriptor'
  } satisfies GPUComputePassDescriptor;


  // TGPU Types
  const Uniforms = struct({
    fun: f32,
    time: f32
  });

  const DiffusionUniforms = struct({
    diffusion_constant: f32,
    dt: f32,
    grid_size: u32
  });

  // Texture creation

  const textureSize = 128;
  let [texture, texture_write, prev_texture] = createFluidTexture(device, textureSize);

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

  const diffusionUniforms = tgpu.createBuffer(DiffusionUniforms, {
    'diffusion_constant': 0.001,
    'dt': 0.03,
    'grid_size': textureSize
  }).$device(device).$usage(tgpu.Uniform);


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
  });


  /*
  @group(0) @binding(0) var density_grid: texture_storage_2d<r32float, write>;
  @group(0) @binding(1) var prev_density_grid: texture_storage_2d<r32float, read>;
  @group(0) @binding(2) var<uniform>uniforms : Uniforms;
*/



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

  function performDiffusion(encoder: GPUCommandEncoder) {
    for (var i = 0; i < 10; i++) {
      const diffusionPass = encoder.beginComputePass(diffusePassDescriptor);

      const diffusionBindGroup = device!.createBindGroup({
        layout: diffusionPipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: texture.createView()
          },
          {
            binding: 1,
            resource: texture_write.createView()
          },
          {
            binding: 2,
            resource: prev_texture.createView()
          },
          {
            binding: 3,
            resource: diffusionUniforms
          }
        ]
      });

      diffusionPass.setPipeline(diffusionPipeline);
      diffusionPass.setBindGroup(0, diffusionBindGroup);
      diffusionPass.dispatchWorkgroups(textureSize / 8, textureSize / 8);
      diffusionPass.end();
      [texture, texture_write] = [texture_write, texture]
    }

    [prev_texture, texture] = [texture, prev_texture]
  }

  function render() {
    resizeCanvasToDisplaySize(canvas);
    renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
    const encoder = device?.createCommandEncoder({ label: 'commandendocoer' });

    // const computePass = encoder?.beginComputePass(computePassDescriptor);
    // computePass?.setPipeline(computePipeline);
    // computePass?.setBindGroup(0, computeBindGroup);
    // tgpu.write(computeUniforms, {
    //   fun: Math.PI,
    //   time: performance.now() / 200
    // })
    // computePass?.dispatchWorkgroups(textureSize / 8, textureSize / 8);
    // computePass?.end();

    performDiffusion(encoder!);

    const renderBindGroup = device!.createBindGroup({
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
