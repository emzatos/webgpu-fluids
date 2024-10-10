export class DensityTextures {
    readTexture: GPUTexture;
    writeTexture: GPUTexture;
    prevTexture: GPUTexture;

    constructor({
        device,
        textureSize,
        data,
    }: {
        device: GPUDevice;
        textureSize: number;
        data?: Float32Array;
    }) {

        const descriptor: GPUTextureDescriptor = {
            label: "storage texture",
            format: "r32float",
            size: [textureSize, textureSize],
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST,
        }

        const textureData = data ?? this.initializeData(textureSize);

        const textures = Array.from({ length: 3 }).map(() => {
            const texture = device.createTexture(descriptor);
            device.queue.writeTexture(
                { texture: texture },
                textureData,
                {
                    bytesPerRow: Float32Array.BYTES_PER_ELEMENT * textureSize,
                    rowsPerImage: textureSize,
                },
                { width: textureSize, height: textureSize },
            );

            return texture;
        });

        this.readTexture = textures[0];
        this.writeTexture = textures[1];
        this.prevTexture = textures[2];

    }

    initializeData(textureSize: number) {
        const N = textureSize - 2;
        const textureData = new Float32Array((N + 2) * (N + 2));
        for (let x = 1; x <= N; ++x) {
            for (let y = 1; y <= N; ++y) {
                const dx = x - (N + 2) / 2;
                const dy = y - (N + 2) / 2;
                const dist = Math.hypot(dx, dy);
                if (dist < 20) {
                    textureData[y * (N + 2) + x] = 1;
                }
            }
        }

        return textureData;
    }

    swap() {
        [this.readTexture, this.writeTexture] = [this.writeTexture, this.readTexture];
    }

    commit() {
        [this.prevTexture, this.writeTexture] = [this.writeTexture, this.prevTexture];
    }
}