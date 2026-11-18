import { DEFAULT_PRESSURE_ITERATIONS, type PressureIterations } from "./FluidSimulation";

export const settings = {
    palette: 0,
    dyeSize: 256,
    pressureIterations: DEFAULT_PRESSURE_ITERATIONS as PressureIterations,
    brushRadius: 256 / 50,
    resetRequested: false,
};