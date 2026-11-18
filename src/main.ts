import "./style.css";
import { tgpu } from "typegpu";
import { attachAutoResizer } from "typegpu/common";
import { MouseHandler } from "./MouseHandler";
import { FluidSimulation } from "./FluidSimulation";
import { Renderer } from "./Renderer";
import { settings } from "./settings";
import { createElement } from "react";
import { createRoot } from "react-dom/client";
import { FluidControls } from "./FluidControls";

const controlsRoot = document.getElementById("fluid-controls");
if (controlsRoot) {
  createRoot(controlsRoot).render(createElement(FluidControls));
}

async function main() {
  const adapter = await navigator.gpu?.requestAdapter();
  const device = await adapter?.requestDevice({
    requiredFeatures: ["float32-filterable"],
  });

  if (!device) {
    alert("WebGPU not supported on this device");
    return;
  }

  const root = tgpu.initFromDevice({ device });
  const canvas = document.querySelector("canvas") as HTMLCanvasElement;
  const context = canvas.getContext("webgpu") as GPUCanvasContext;
  const textureSize = 256;
  attachAutoResizer({ root, canvas });

  const simulation = new FluidSimulation(root, textureSize);
  const mouseHandler = new MouseHandler(canvas, textureSize);
  const renderer = new Renderer(root, context);

  function render() {
    const frame = root.createCommandEncoder({
      label: "fluid frame",
    });
    const encoder = root.unwrap(frame);

    if (mouseHandler.mouseDown) {
      simulation.splat(
        encoder,
        mouseHandler.mousePosition.x,
        mouseHandler.mousePosition.y,
        (mouseHandler.mousePosition.x - mouseHandler.lastMousePosition.x) * 0.1,
        (mouseHandler.mousePosition.y - mouseHandler.lastMousePosition.y) * 0.1,
        settings.brushRadius,
        1
      );
    }

    simulation.update(encoder);
    renderer.render(encoder, simulation.getDensityTexture().createView(), settings.palette);

    frame.submit();

    mouseHandler.lastMousePosition = {
      x: mouseHandler.mousePosition.x,
      y: mouseHandler.mousePosition.y,
    };
    requestAnimationFrame(render);
  }
  render();
}

main();
