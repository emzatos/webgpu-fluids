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

    this.canvas.addEventListener("mouseup", this.onMouseUp);
  }

  private onMouseMove = (e: MouseEvent) => {
    const rect = this.canvas.getBoundingClientRect();
    this.mousePosition.x = (e.clientX / rect.width) * this.textureSize;
    this.mousePosition.y =
      this.textureSize - (e.clientY / rect.height) * this.textureSize;
  };

  private onMouseDown = () => {
    this.mouseDown = true;
  };

  private onMouseUp = () => {
    this.mouseDown = false;
  };
}
