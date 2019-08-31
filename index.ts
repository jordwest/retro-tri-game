const GL = WebGLRenderingContext;

namespace Result {
  export type Ok<A> = { type: "ok"; value: A };
  export type Err<B> = { type: "err"; error: B };
  export type T<A, B> = Ok<A> | Err<B>;

  export const isOk = <A, B>(t: T<A, B>): t is Ok<A> => t.type === "ok";
  export const isErr = <A, B>(t: T<A, B>): t is Err<B> => t.type === "err";
  export const ok = <A, B>(value: A): T<A, B> => ({ type: "ok", value });
  export const err = <A, B>(error: B): T<A, B> => ({ type: "err", error });
  export const unwrap = <A, B>(v: T<A, B>): A => {
    if (v.type === "err") {
      console.error(v.error);
      throw v.error;
    }
    return v.value;
  };
  export const map = <A, B, C>(t: T<A, B>, f: (v: A) => T<C, B>) =>
    isOk(t) ? f(t.value) : t;
  export const mapErr = <A, B, C>(t: T<A, B>, f: (v: B) => T<C, B>) =>
    isErr(t) ? f(t.error) : t;
}

namespace Game {
  export namespace Shader {
    export type Source = string & { __shaderSrc: never };
    export type ShaderError =
      | { type: "compile_error"; shaderType: "frag" | "vert"; e: string }
      | { type: "link_error"; e: string };

    export namespace Utils {
      export function createProgram(
        gl: WebGL.Ctx,
        vertSrc: Source,
        fragSrc: Source
      ): Result.T<WebGLProgram, ShaderError> {
        const compileShader = (
          shaderType: "frag" | "vert",
          source: Source
        ): Result.T<WebGLShader, ShaderError> => {
          const shader = gl.createShader(
            shaderType === "vert" ? gl.VERTEX_SHADER : gl.FRAGMENT_SHADER
          );
          gl.shaderSource(shader, source);
          gl.compileShader(shader);
          if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            return Result.ok(shader);
          }
          return Result.err<WebGLProgram, ShaderError>({
            type: "compile_error",
            shaderType,
            e: gl.getShaderInfoLog(shader)
          });
        };

        const vert = compileShader("vert", vertSrc);
        const frag = compileShader("frag", fragSrc);

        return Result.map(vert, vertOk =>
          Result.map(frag, fragOk => {
            const program = gl.createProgram();
            gl.attachShader(program, vertOk);
            gl.attachShader(program, fragOk);
            gl.linkProgram(program);
            if (gl.getProgramParameter(program, gl.LINK_STATUS)) {
              Result.err<WebGLProgram, ShaderError>({
                type: "link_error",
                e: gl.getProgramInfoLog(program)
              });
            }
            return Result.ok(program);
          })
        );
      }
    }

    export namespace Color {
      export const vert = `
                precision mediump float;
                attribute vec2 position;
                uniform float uScale;

                void main() {
                    gl_Position = vec4(position * uScale, 0.0, 1.0);
                }
            ` as Shader.Source;

      export const frag = `
                precision mediump float;
                void main() {
                    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
                }
            ` as Shader.Source;
    }

    export namespace Glow {
      export const vert = `
                precision mediump float;
                attribute vec2 position;
                attribute vec2 aTexCoord;
                varying vec2 texCoord;

                void main() {
                    texCoord = aTexCoord;
                    gl_Position = vec4(position, 0.0, 1.0);
                }
            ` as Shader.Source;

      export const frag = `
                precision mediump float;
                uniform sampler2D uSourceImage;
                uniform vec2 uResolution;
                varying vec2 texCoord;

                void main() {
                  const int kernelHalfWidth = 12;

                  vec3 srcCol = texture2D(uSourceImage, texCoord).xyz;
                  vec3 glowCol = vec3(0.0, 0.0, 0.0);
                  for (int x = -kernelHalfWidth; x < kernelHalfWidth; x++) {
                    for (int y = -kernelHalfWidth; y < kernelHalfWidth; y++) {
                      vec2 perPixel = 1.0 / uResolution;
                      vec2 offset = vec2(perPixel.x * float(x), perPixel.y * float(y));
                      // float dist = sqrt(exp(offset.x, 2), exp(offset.y, 2));
                      float dist = distance(vec2(0.0, 0.0), offset);
                      float magnitude = (1.0 - (dist / float(kernelHalfWidth))) + 0.5;
                      glowCol += (texture2D(uSourceImage, texCoord + offset).xyz * magnitude);
                    }
                  }
                  glowCol = glowCol / ((float(kernelHalfWidth) * 2.0) * (float(kernelHalfWidth) * 2.0));
                  gl_FragColor = vec4(srcCol, 1.0) + vec4(glowCol, 1.0);
                  // gl_FragColor = vec4(srcCol, 1.0);
                  // gl_FragColor = vec4(glowCol, 1.0);
                }
        ` as Shader.Source;
    }
  }

  export namespace Vec2 {
    export type T = {
      x: number;
      y: number;
    };

    export class FloatArray {
      public arr: Float32Array;
      constructor(size: number) {
        this.arr = new Float32Array(size * 2);
      }
      get length(): number {
        return this.arr.length / 2;
      }
      get(index: number): T {
        return { x: this.arr[index * 2], y: this.arr[index * 2 + 1] };
      }
      set(index: number, v: T) {
        this.arr[index * 2] = v.x;
        this.arr[index * 2 + 1] = v.y;
      }
      static from(elements: Vec2.T[]): FloatArray {
        const self = new FloatArray(elements.length);
        elements.forEach((el, i) => {
          self.set(i, el);
        });
        return self;
      }
    }
  }

  export namespace WebGL {
    export type Ctx = WebGLRenderingContext;
    export function createQuad(x1: number, y1: number, x2: number, y2: number) {
      return Vec2.FloatArray.from([
        { x: x1, y: y1 },
        { x: x2, y: y1 },
        { x: x1, y: y2 },
        { x: x2, y: y1 },
        { x: x1, y: y2 },
        { x: x2, y: y2 }
      ]);
    }

    export class RenderTarget {
      framebuffer: WebGLFramebuffer;
      texture: WebGLTexture;
      gl: Ctx;

      private constructor(gl: Ctx, width: number, height: number) {
        this.framebuffer = gl.createFramebuffer();
        if (!this.framebuffer) {
          throw new Error("Could not create framebuffer");
        }
        this.texture = gl.createTexture();
        if (!this.texture) {
          throw new Error("Could not create texture");
        }
        this.gl = gl;

        gl.bindTexture(gl.TEXTURE_2D, this.texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

        gl.getExtension("OES_texture_float");
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.RGBA,
          width,
          height,
          0,
          gl.RGBA,
          gl.FLOAT,
          null
        );

        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
        // Attach a texture to it.
        gl.framebufferTexture2D(
          gl.FRAMEBUFFER,
          gl.COLOR_ATTACHMENT0,
          gl.TEXTURE_2D,
          this.texture,
          0
        );
      }

      static create(gl: Ctx, width: number, height: number) {
        return new RenderTarget(gl, width, height);
      }

      use() {
        this.gl.bindFramebuffer(GL.FRAMEBUFFER, this.framebuffer);
      }

      disconnect() {
        this.gl.bindFramebuffer(GL.FRAMEBUFFER, null);
      }
    }
  }

  export function main() {
    const canvas = document.getElementById("game") as HTMLCanvasElement;

    canvas.width = window.devicePixelRatio * window.innerWidth;
    canvas.height = window.devicePixelRatio * window.innerHeight;

    const gl = canvas.getContext("webgl");
    const program = Result.unwrap(
      Shader.Utils.createProgram(gl, Shader.Color.vert, Shader.Color.frag)
    );

    const glowProgram = Result.unwrap(
      Shader.Utils.createProgram(gl, Shader.Glow.vert, Shader.Glow.frag)
    );

    const triBuffer = gl.createBuffer();
    const tri = Vec2.FloatArray.from([
      { x: -1, y: -1 },
      { x: 0, y: 1 },
      { x: 1, y: -1 }
    ]);

    const firstPassTarget = WebGL.RenderTarget.create(
      gl,
      canvas.width,
      canvas.height
    );
    firstPassTarget.use();

    gl.bindBuffer(gl.ARRAY_BUFFER, triBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, tri.arr, gl.STATIC_DRAW);

    const attribLocation = gl.getAttribLocation(program, "position");
    gl.vertexAttribPointer(attribLocation, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(attribLocation);

    gl.useProgram(program);

    const uScale = gl.getUniformLocation(program, "uScale");
    gl.uniform1f(uScale, 0.2);

    // FIRST PASS - Raw game

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.drawArrays(gl.TRIANGLES, 0, tri.length);

    firstPassTarget.disconnect();

    // SECOND PASS - Glow effect

    const quadBuffer = gl.createBuffer();
    const quad = WebGL.createQuad(-1, -1, 1, 1);

    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, quad.arr, gl.STATIC_DRAW);

    gl.useProgram(glowProgram);

    const quadAttribLocation = gl.getAttribLocation(glowProgram, "position");
    gl.vertexAttribPointer(quadAttribLocation, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(quadAttribLocation);

    const texQuadBuffer = gl.createBuffer();
    const texQuad = WebGL.createQuad(0, 0, 1, 1);
    gl.bindBuffer(gl.ARRAY_BUFFER, texQuadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texQuad.arr, gl.STATIC_DRAW);

    const texQuadAttribLocation = gl.getAttribLocation(
      glowProgram,
      "aTexCoord"
    );
    gl.vertexAttribPointer(texQuadAttribLocation, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(texQuadAttribLocation);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, firstPassTarget.texture);
    const uTexLoc = gl.getUniformLocation(glowProgram, "uSourceImage");
    gl.uniform1i(uTexLoc, 0);

    const uResolution = gl.getUniformLocation(glowProgram, "uResolution");
    gl.uniform2f(uResolution, canvas.width, canvas.height);

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.drawArrays(gl.TRIANGLES, 0, quad.length);
  }
}

Game.main();
