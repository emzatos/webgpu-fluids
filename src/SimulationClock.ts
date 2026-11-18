const STEP_MS = 1000 / 60;
export const MAX_STEPS_PER_FRAME = 3;

export class SimulationClock {
  private lastTime?: number;
  private accumulated = 0;

  advance(timestamp: number): number {
    const previous = this.lastTime;
    this.lastTime = timestamp;
    if (previous === undefined) return 1;

    // Bound catch-up work after a stall or when returning to the tab.
    this.accumulated += Math.min(timestamp - previous, MAX_STEPS_PER_FRAME * STEP_MS);
    const steps = Math.floor((this.accumulated + 1e-6) / STEP_MS);
    this.accumulated = Math.max(0, this.accumulated - steps * STEP_MS);
    return steps;
  }
}
