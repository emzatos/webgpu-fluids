import "./style.css";
import { MouseHandler } from "./MouseHandler";
import { FluidSimulation } from "./FluidSimulation";
import { Renderer } from "./Renderer";

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
  const textureSize = 1024;

  const simulation = new FluidSimulation(device, textureSize);
  const mouseHandler = new MouseHandler(canvas, textureSize);
  const renderer = new Renderer(
    device,
    canvas,
    context,
    simulation,
    mouseHandler,
    textureSize
  );

  renderer.render();
}

main();
