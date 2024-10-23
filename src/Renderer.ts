import { f32, struct } from "typegpu/data/index";
import shaders from "./shaders.wgsl?raw";
import tgpu from "typegpu";
import { FluidSimulation } from "./FluidSimulation";
import { MouseHandler } from "./MouseHandler";

export class Renderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private canvas: HTMLCanvasElement;
  private presentationFormat: GPUTextureFormat;
  private textureSize: number;
  private renderPipeline!: GPURenderPipeline;
  private renderPassDescriptor!: GPURenderPassDescriptor & {
    colorAttachments: [GPURenderPassColorAttachment];
  };
  private sampler!: GPUSampler;
  private fragmentUniforms: any; // Using any for tgpu type
  private canvasToSizeMap: WeakMap<
    HTMLCanvasElement,
    { width: number; height: number }
  >;
  private simulation: FluidSimulation;
  private mouseHandler: MouseHandler;
  private observer!: ResizeObserver;

  constructor(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    simulation: FluidSimulation,
    mouseHandler: MouseHandler,
    textureSize: number
  ) {
    this.device = device;
    this.canvas = canvas;
    this.context = context;
    this.simulation = simulation;
    this.mouseHandler = mouseHandler;
    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    this.canvasToSizeMap = new WeakMap();
    this.textureSize = textureSize;

    this.initializeRenderer(shaders);
    this.setupResizeObserver();
  }

  private initializeRenderer(shaders: string): void {
    // Configure context
    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
    });

    // Create shader module and pipeline
    const renderModule = this.device.createShaderModule({
      label: "Shaders for triangle",
      code: shaders,
    });

    this.renderPipeline = this.device.createRenderPipeline({
      label: "render pipeline",
      layout: "auto",
      vertex: {
        module: renderModule,
      },
      fragment: {
        module: renderModule,
        targets: [{ format: this.presentationFormat }],
      },
    });

    // Create render pass descriptor
    this.renderPassDescriptor = {
      label: "render pass descriptor",
      colorAttachments: [
        {
          view: {} as GPUTextureView,
          clearValue: [0.3, 0.3, 0.3, 1.0],
          loadOp: "clear",
          storeOp: "store",
        } as GPURenderPassColorAttachment,
      ],
    };

    // Create sampler and uniforms
    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });

    const FragmentUniforms = struct({
      time: f32,
    });

    this.fragmentUniforms = tgpu
      .createBuffer(FragmentUniforms)
      .$device(this.device)
      .$usage(tgpu.Uniform);
  }

  private resizeCanvasToDisplaySize(): boolean {
    const dimensions = this.canvasToSizeMap.get(this.canvas) || this.canvas;
    const width = Math.max(
      1,
      Math.min(dimensions.width, this.device.limits.maxTextureDimension2D)
    );
    const height = Math.max(
      1,
      Math.min(dimensions.height, this.device.limits.maxTextureDimension2D)
    );

    const needResize =
      this.canvas.width !== width || this.canvas.height !== height;
    if (needResize) {
      this.canvas.width = width;
      this.canvas.height = height;
    }

    return needResize;
  }

  private setupResizeObserver(): void {
    this.observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        this.canvasToSizeMap.set(entry.target as HTMLCanvasElement, {
          width: entry.contentBoxSize[0].inlineSize,
          height: entry.contentBoxSize[0].blockSize,
        });
      }
      this.render();
    });

    this.observer.observe(this.canvas);
  }

  public render = (): void => {
    this.resizeCanvasToDisplaySize();
    // Update render pass descriptor with current texture view
    this.renderPassDescriptor.colorAttachments[0].view = this.context
      .getCurrentTexture()
      .createView();

    const encoder = this.device.createCommandEncoder({
      label: "command encoder",
    });

    // Handle mouse interaction
    if (this.mouseHandler.mouseDown) {
      this.simulation.splat(
        encoder,
        this.mouseHandler.mousePosition.x,
        this.mouseHandler.mousePosition.y,
        (this.mouseHandler.mousePosition.x -
          this.mouseHandler.lastMousePosition.x) *
          0.1,
        (this.mouseHandler.mousePosition.y -
          this.mouseHandler.lastMousePosition.y) *
          0.1,
        this.textureSize / 50,
        1
      );
    }

    // Update simulation and uniforms
    this.simulation.update(encoder);
    this.fragmentUniforms.write({ time: performance.now() / 2000 });

    // Create bind group for rendering
    const renderBindGroup = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.simulation.getDensityTexture().createView(),
        },
        {
          binding: 1,
          resource: this.sampler,
        },
        {
          binding: 2,
          resource: this.fragmentUniforms,
        },
      ],
    });

    // Execute render pass
    const pass = encoder.beginRenderPass(this.renderPassDescriptor);
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, renderBindGroup);
    pass.draw(6);
    pass.end();

    // Submit command buffer
    const commandBuffer = encoder.finish();
    this.device.queue.submit([commandBuffer]);

    // Update mouse position
    this.mouseHandler.lastMousePosition = {
      x: this.mouseHandler.mousePosition.x,
      y: this.mouseHandler.mousePosition.y,
    };

    requestAnimationFrame(this.render);
  };

  public cleanup(): void {
    this.observer.disconnect();
  }
}
