import { useControls, Leva } from "leva";
import { settings } from "./settings";
import { createElement } from "react";

export function FluidControls() {
    useControls({
        "Color scheme": {
            value: settings.palette,
            options: { Rainbow: 0, Ember: 3, "White on black": 1, "Black on white": 2 },
            onChange: (value: number) => { settings.palette = value; },
        },
        "Dye resolution (resets fluid)": {
            value: settings.dyeSize,
            options: { "256 × 256": 256, "512 × 512": 512 },
            onChange: (value: number) => { settings.dyeSize = value; },
        },
        "Brush size": {
            value: settings.brushRadius,
            min: 1,
            max: 40,
            step: 0.01,
            onChange: (value: number) => { settings.brushRadius = value; },
        },
    });
    return createElement(Leva)
}