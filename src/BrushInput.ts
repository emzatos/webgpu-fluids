export interface BrushPoint {
  x: number;
  y: number;
}

export class BrushInput {
  private pointerId?: number;
  private previous: BrushPoint = { x: 0, y: 0 };
  private current: BrushPoint = this.previous;
  private pending = false;

  constructor(private readonly canvas: HTMLCanvasElement, private readonly textureSize: number) {
    canvas.addEventListener("pointerdown", this.onPointerDown);
    canvas.addEventListener("pointermove", this.onPointerMove);
    canvas.addEventListener("pointerup", this.onPointerUp);
    canvas.addEventListener("lostpointercapture", this.onLostCapture);
  }

  takeStroke() {
    if (this.pointerId === undefined && !this.pending) return undefined;
    const stroke = { start: this.previous, end: this.current };
    this.previous = this.current;
    this.pending = false;
    return stroke;
  }

  private point(event: PointerEvent): BrushPoint {
    const { left, top, width, height } = this.canvas.getBoundingClientRect();
    const toCell = (fraction: number) => 1 + Math.max(0, Math.min(1, fraction)) * (this.textureSize - 1);
    return {
      x: toCell((event.clientX - left) / width),
      y: toCell(1 - (event.clientY - top) / height),
    };
  }

  private onPointerDown = (event: PointerEvent) => {
    if (this.pointerId !== undefined || event.button !== 0) return;
    this.pointerId = event.pointerId;
    this.canvas.setPointerCapture(event.pointerId);
    this.previous = this.current = this.point(event);
    this.pending = true;
  };

  private onPointerMove = (event: PointerEvent) => {
    if (event.pointerId !== this.pointerId) return;
    this.current = this.point(event);
    this.pending = true;
  };

  private onPointerUp = (event: PointerEvent) => {
    if (event.pointerId !== this.pointerId) return;
    const end = this.point(event);
    this.pending ||= end.x !== this.current.x || end.y !== this.current.y;
    this.current = end;
    this.pointerId = undefined;
  };

  private onLostCapture = (event: PointerEvent) => {
    if (event.pointerId === this.pointerId) this.pointerId = undefined;
  };
}
