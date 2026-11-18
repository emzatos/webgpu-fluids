export class MouseHandler {
  private canvas: HTMLCanvasElement;
  private textureSize: number;
  public mousePosition: { x: number; y: number };
  public lastMousePosition: { x: number; y: number };
  public mouseDown: boolean;

  constructor(canvas: HTMLCanvasElement, textureSize: number) {
    this.canvas = canvas;
    this.textureSize = textureSize;
    this.mousePosition = { x: 0, y: 0 };
    this.lastMousePosition = { x: 0, y: 0 };
    this.mouseDown = false;
    this.addEventListeners();
  }

  private addEventListeners() {
    this.canvas.addEventListener("mousemove", this.onMouseMove);

    this.canvas.addEventListener("mousedown", this.onMouseDown);

    window.addEventListener("mouseup", this.onMouseUp);
  }

  private onMouseMove = (e: MouseEvent) => {
    const rect = this.canvas.getBoundingClientRect();
    const span = this.textureSize - 1;
    this.mousePosition.x = 1 + ((e.clientX - rect.left) / rect.width) * span;
    this.mousePosition.y = 1 + (1 - (e.clientY - rect.top) / rect.height) * span;
  };

  private onMouseDown = (e: MouseEvent) => {
    this.onMouseMove(e);
    this.lastMousePosition = { ...this.mousePosition };
    this.mouseDown = true;
  };

  private onMouseUp = () => {
    this.mouseDown = false;
  };
}
