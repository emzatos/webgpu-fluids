import { expect, test } from "bun:test";
import { MAX_STEPS_PER_FRAME, SimulationClock } from "../src/SimulationClock";

test("60 Hz and 120 Hz displays advance the same number of simulation steps", () => {
  const simulate = (refresh: number) => {
    const clock = new SimulationClock();
    let steps = 0;
    for (let frame = 0; frame < refresh; frame++) {
      steps += clock.advance(frame * 1000 / refresh);
    }
    return steps;
  };
  expect(simulate(60)).toBe(60);
  expect(simulate(120)).toBe(60);
});

test("the clock caps catch-up after a long stall and resumes on time", () => {
  const clock = new SimulationClock();
  expect(clock.advance(0)).toBe(1);
  expect(clock.advance(5000)).toBe(MAX_STEPS_PER_FRAME);
  expect(clock.advance(5004)).toBe(0);
  expect(clock.advance(5017)).toBe(1);
});
