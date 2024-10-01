import shaders from "./shaders.wgsl?raw";
import './style.css'

const rand = (min?: number | undefined, max?: number | undefined) => {
  if (min === undefined) {
    min = 0;
    max = 1;
  } else if (max === undefined) {
    max = min;
    min = 0;
  }
  return min + Math.random() * (max-min);
}

function createCircleVertices({
  radius = 1,
  numSubdivisions = 24,
  innerRadius = 0,
  startAngle = 0,
  endAngle = Math.PI * 2,
} = {}) {
  const numVertices = numSubdivisions * 3 * 2;
  const vertexData = new Float32Array(numVertices * 2);

  let offset = 0;
  const addVertex = (x, y) => {
    vertexData[offset++] = x;
    vertexData[offset++] = y;
  };

  for (let i=0; i < numSubdivisions; i++) {
    const angle1 = startAngle + (i + 0) * (endAngle - startAngle) / numSubdivisions;
    const angle2 = startAngle + (i + 1) * (endAngle - startAngle) / numSubdivisions;

    const c1 = Math.cos(angle1);
    const s1 = Math.sin(angle1);
    const c2 = Math.cos(angle2);
    const s2 = Math.sin(angle2);

    addVertex(radius * c1, radius * s1);
    addVertex(radius * c2, radius * s2);
    addVertex(c1 * innerRadius, s1 * innerRadius);

    addVertex(innerRadius * c1, innerRadius * s1);
    addVertex(innerRadius * c2, innerRadius * s2);
    addVertex(radius * c2, radius * s2);
  }

  return {
    vertexData,
    numVertices
  }
}

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

  const {vertexData, numVertices} = createCircleVertices({
    radius: 0.5,
    innerRadius: 0.25
  });

  const vertexStorageBuffer = device.createBuffer({
    label: "vertext storage bungegr",
    size: vertexData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(vertexStorageBuffer, 0, vertexData);

  const uniformBufferSize = 4 * 4 + 2 * 4 + 2 * 4;

  const kColorOffset = 0;
  const kScaleOffset = 0;
  const kOffsetOffset = 4;
  const kNumObject = 100;
  const objectInfos: { scale: number; }[] = [];

  const staticUnitSize = 4 * 4 + 4 * 2 + 2 * 4; // color 4d and offset 2d
  const changingUnitSize = 4 * 2; // scale 2d
  const staticStorageBufferSize = staticUnitSize * kNumObject;
  const changingStorageBufferSize = changingUnitSize * kNumObject;

  const staticStorageBuffer = device.createBuffer({
    label: 'static storage',
    size: staticStorageBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });

  const changingStorageBuffer = device.createBuffer({
    label: 'chanign storage buffer',
    size: changingStorageBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });

  {
    const staticStorageValues = new Float32Array(staticStorageBufferSize / 4);
    for (let i=0; i < kNumObject; i++) {
      let staticOffset = i * (staticUnitSize / 4)
      staticStorageValues.set([rand(), rand(), rand(), 1], staticOffset + kColorOffset);
      staticStorageValues.set([rand(-0.9, 0.9), rand(-0.9, 0.9)], staticOffset + kOffsetOffset);

      objectInfos.push({
        scale: rand(0.2, 0.5)
      })
    }
    device.queue.writeBuffer(staticStorageBuffer, 0, staticStorageValues);
  }

  const storageValues = new Float32Array(changingStorageBufferSize / 4);
  
  const bindGroup = device.createBindGroup({
    label: 'bind group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0, resource: { buffer: staticStorageBuffer}
      },
      {
        binding: 1, resource: { buffer: changingStorageBuffer}
      },
      {
        binding: 2, resource: { buffer: vertexStorageBuffer}
      }
    ]
  })


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
    const aspect = canvas.width / canvas.height;
    objectInfos.forEach(({scale}, ndx) => {
      const offset = ndx * (changingUnitSize / 4);
      storageValues.set([scale / aspect, scale], offset + kScaleOffset);
    });
    device?.queue.writeBuffer(changingStorageBuffer, 0, storageValues);

    pass?.setBindGroup(0, bindGroup);
    pass?.draw(numVertices, kNumObject);

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
