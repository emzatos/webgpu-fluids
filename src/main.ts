import "./style.css";
import { tgpu } from "typegpu";
import { attachAutoResizer } from "typegpu/common";
import { BrushInput, type BrushPoint } from "./BrushInput";
import { SimulationClock } from "./SimulationClock";
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

function interpolate(start: BrushPoint, end: BrushPoint, fraction: number): BrushPoint {
  return {
    x: start.x + (end.x - start.x) * fraction,
    y: start.y + (end.y - start.y) * fraction,
  };
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
  const context = root.configureContext({ canvas });
  const textureSize = 256;
  attachAutoResizer({ root, canvas });

  let dyeSize = settings.dyeSize;
  let simulation = new FluidSimulation(root, textureSize, { dyeSize });
  const brushInput = new BrushInput(canvas, textureSize);
  const renderer = new Renderer(root, context);

  const clock = new SimulationClock();

  function render(timestamp: number) {
    if (settings.dyeSize !== dyeSize) {
      dyeSize = settings.dyeSize;
      simulation.destroy();
      simulation = new FluidSimulation(root, textureSize, { dyeSize });
    }

    if (settings.resetRequested) {
      settings.resetRequested = false;
      simulation.destroy();
      simulation = new FluidSimulation(root, textureSize, { dyeSize });
    }

    const frame = root.createCommandEncoder({
      label: "fluid frame",
    });
    const encoder = root.unwrap(frame);

    const steps = clock.advance(timestamp);
    const stroke = steps > 0 ? brushInput.takeStroke() : undefined;
    for (let step = 0; step < steps; step++) {
      if (stroke) {
        simulation.splat(
          encoder,
          interpolate(stroke.start, stroke.end, step / steps),
          interpolate(stroke.start, stroke.end, (step + 1) / steps),
          settings.brushRadius, 1, step
        );
      }
      simulation.update(encoder, settings.pressureIterations);
    }
    renderer.render(frame, simulation.getDensityView(), settings.palette);

    frame.submit();

    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
}

main();
