import { tgpu } from "typegpu";
import { FluidSimulation, type DyeTransport } from "../src/FluidSimulation";
import { Renderer } from "../src/Renderer";

const GRID = 256;
const WARMUP_STEPS = 8;
const STROKE_STEPS = 48;
const SETTLE_STEPS = 16;

const strokes = {
  line: (t: number) => [0.12 + 0.76 * t, 0.30 + 0.12 * t],
  spiral: (t: number) => {
    const angle = 3 * Math.PI * t;
    const radius = 0.06 + 0.32 * t;
    return [0.5 + radius * Math.cos(angle), 0.5 + radius * Math.sin(angle)];
  },
  wall: (t: number) => [0.05 - 0.048 * t, 0.13 + 0.74 * t],
};

const canvas = document.querySelector("canvas")!;
const results = document.querySelector<HTMLPreElement>("#results")!;
const runButton = document.querySelector<HTMLButtonElement>("#run")!;
const scene = document.querySelector<HTMLSelectElement>("#scene")!;
const transport = document.querySelector<HTMLSelectElement>("#transport")!;
const diffusion = document.querySelector<HTMLSelectElement>("#diffusion")!;

function percentile(values: number[], fraction: number) {
  return [...values].sort((a, b) => a - b)[Math.ceil(values.length * fraction) - 1].toFixed(2);
}

async function readFields(device: GPUDevice, textures: GPUTexture[]) {
  const encoder = device.createCommandEncoder();
  const copies = textures.map((texture) => {
    const bytesPerRow = Math.ceil(texture.width * 4 / 256) * 256;
    const buffer = device.createBuffer({
      size: bytesPerRow * texture.height,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, [texture.width, texture.height]);
    return { buffer, stride: bytesPerRow / 4 };
  });
  device.queue.submit([encoder.finish()]);
  return Promise.all(copies.map(async ({ buffer, stride }) => {
    try {
      await buffer.mapAsync(GPUMapMode.READ);
      return { values: new Float32Array(buffer.getMappedRange().slice(0)), stride };
    } finally {
      buffer.destroy();
    }
  }));
}

runButton.addEventListener("click", async () => {
  runButton.disabled = true;
  scene.disabled = true;
  transport.disabled = true;
  diffusion.disabled = true;
  results.textContent = "Running stroke replay…";
  let device: GPUDevice | undefined;

  try {
    const root = await tgpu.init({ device: { requiredFeatures: ["float32-filterable"] } });
    device = root.device;
    device.pushErrorScope("validation");
    const simulation = new FluidSimulation(root, GRID, {
      startEmpty: true,
      dyeTransport: transport.value as DyeTransport,
      diffusionIterations: Number(diffusion.value),
    });
    const renderer = new Renderer(root, root.configureContext({ canvas }));
    const stroke = strokes[scene.value as keyof typeof strokes];
    const position = (t: number) => stroke(t).map((coordinate) => 1 + coordinate * (GRID - 1));
    const encoding: number[] = [];
    const completion: number[] = [];

    for (let step = -WARMUP_STEPS; step < STROKE_STEPS + SETTLE_STEPS; step++) {
      await new Promise(requestAnimationFrame);
      const start = performance.now();
      const frame = root.createCommandEncoder();
      const encoder = root.unwrap(frame);
      if (step >= 0 && step < STROKE_STEPS) {
        const [x, y] = position(step / (STROKE_STEPS - 1));
        const [previousX, previousY] = position(Math.max(0, step - 1) / (STROKE_STEPS - 1));
        simulation.splat(encoder, x, y, (x - previousX) * 0.1, (y - previousY) * 0.1, 3, 1);
      }
      simulation.update(encoder);
      renderer.render(frame, simulation.getDensityView(), 1);
      frame.submit();
      const submitted = performance.now();
      await device.queue.onSubmittedWorkDone();
      if (step >= 0) {
        encoding.push(submitted - start);
        completion.push(performance.now() - submitted);
      }
    }

    const [dye, horizontal, vertical] = await readFields(device, [
      simulation.getDensityTexture(),
      simulation.getVelXTexture(),
      simulation.getVelYTexture(),
    ]);
    const error = await device.popErrorScope();
    if (error) throw error;

    let mass = 0;
    let peak = 0;
    let min = 0;
    let gradient = 0;
    let divergenceSquares = 0;
    for (let y = 1; y <= GRID; y++) {
      for (let x = 1; x <= GRID; x++) {
        const value = dye.values[y * dye.stride + x];
        mass += value;
        peak = Math.max(peak, value);
        min = Math.min(min, value);
        if (x < GRID) gradient += Math.abs(dye.values[y * dye.stride + x + 1] - value);
        if (y < GRID) gradient += Math.abs(dye.values[(y + 1) * dye.stride + x] - value);
        const u = y * horizontal.stride + x;
        const v = y * vertical.stride + x;
        const divergence = (horizontal.values[u + 1] - horizontal.values[u - 1]
          + vertical.values[v + vertical.stride] - vertical.values[v - vertical.stride]) * GRID / 2;
        divergenceSquares += divergence ** 2;
      }
    }

    if (![dye, horizontal, vertical].every(({ values }) => values.every(Number.isFinite))
        || min < -1e-5 || mass <= 0 || gradient <= 0) {
      throw Error("Invalid output. Expected finite velocity and non-negative, nonuniform dye.");
    }

    results.textContent = [
      "PASS · WebGPU validation and field checks",
      `${encoding.length} measured frames · ${GRID} × ${GRID} grid · one frame in flight`,
      `CPU encode + submit: ${percentile(encoding, 0.5)} ms median · ${percentile(encoding, 0.95)} ms p95`,
      `Submit to completion: ${percentile(completion, 0.5)} ms median · ${percentile(completion, 0.95)} ms p95`,
      `Dye mass: ${mass.toFixed(3)} · peak: ${peak.toFixed(4)}`,
      `Dye edge variation / mass: ${(gradient / mass).toFixed(5)}`,
      `RMS velocity divergence: ${Math.sqrt(divergenceSquares / GRID ** 2).toFixed(4)}`,
    ].join("\n");
  } catch (error) {
    results.textContent = `FAIL · ${error instanceof Error ? error.message : String(error)}`;
  } finally {
    runButton.disabled = false;
    scene.disabled = false;
    transport.disabled = false;
    diffusion.disabled = false;
    // Allow the last submitted image to be presented before releasing test resources.
    if (device) {
      const finishedDevice = device;
      requestAnimationFrame(() => finishedDevice.destroy());
    }
  }
});
