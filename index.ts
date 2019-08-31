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

    namespace GlType {
      export type T = "float" | "int";
      export const toGL = (t: T) => (t === "float" ? GL.FLOAT : GL.INT);
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

    export class Buffer {
      gl: Ctx;
      glBuffer: WebGLBuffer;
      glType: GlType.T;
      length: number;
      componentSize: number;

      constructor(gl: Ctx, glType: GlType.T, componentSize: number) {
        this.gl = gl;
        this.glBuffer = gl.createBuffer();
        this.glType = glType;
        this.componentSize = componentSize;
        this.length = 0;
      }

      set(arr: BufferSource & { length: number }) {
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.glBuffer);
        this.gl.bufferData(
          this.gl.ARRAY_BUFFER,
          arr as any,
          this.gl.STATIC_DRAW
        );
        this.length = arr.length;
      }

      get components() {
        return this.length / this.componentSize;
      }
    }

    export class Uniform {
      gl: Ctx;
      loc: WebGLUniformLocation;

      constructor(gl: Ctx, loc: WebGLUniformLocation) {
        this.gl = gl;
        this.loc = loc;
      }

      set1f(x: number) {
        this.gl.uniform1f(this.loc, x);
      }

      set1i(x: number) {
        this.gl.uniform1i(this.loc, x);
      }

      set2f(x: number, y: number) {
        this.gl.uniform2f(this.loc, x, y);
      }
    }

    export class Program {
      constructor(public gl: Ctx, public glProgram: WebGLProgram) {}

      use() {
        this.gl.useProgram(this.glProgram);
      }

      static create(
        gl: Ctx,
        vertSource: Shader.Source,
        fragSource: Shader.Source
      ) {
        const program = Result.unwrap(
          Shader.Utils.createProgram(gl, vertSource, fragSource)
        );
        return new Program(gl, program);
      }

      addVertexAttribArray(attribName: string, buf: Buffer) {
        this.use();
        const location = this.gl.getAttribLocation(this.glProgram, attribName);
        this.gl.vertexAttribPointer(
          location,
          buf.componentSize,
          GlType.toGL(buf.glType),
          false,
          0,
          0
        );
        this.gl.enableVertexAttribArray(location);
      }

      getUniform(name: string) {
        this.use();
        const uniformLoc = this.gl.getUniformLocation(this.glProgram, name);
        return new Uniform(this.gl, uniformLoc);
      }
    }
  }

  export function main() {
    const canvas = document.getElementById("game") as HTMLCanvasElement;

    canvas.width = window.devicePixelRatio * window.innerWidth;
    canvas.height = window.devicePixelRatio * window.innerHeight;

    const gl = canvas.getContext("webgl");

    const triProgram = WebGL.Program.create(
      gl,
      Shader.Color.vert,
      Shader.Color.frag
    );
    const glowProgram = WebGL.Program.create(
      gl,
      Shader.Glow.vert,
      Shader.Glow.frag
    );

    const triBuffer = new WebGL.Buffer(gl, "float", 2);
    triBuffer.set(
      Vec2.FloatArray.from([{ x: -1, y: -1 }, { x: 0, y: 1 }, { x: 1, y: -1 }])
        .arr
    );

    const firstPassTarget = WebGL.RenderTarget.create(
      gl,
      canvas.width,
      canvas.height
    );
    firstPassTarget.use();

    triProgram.addVertexAttribArray("position", triBuffer);
    triProgram.use();

    const uScale = triProgram.getUniform("uScale");
    uScale.set1f(0.2);

    // FIRST PASS - Raw game

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.drawArrays(gl.TRIANGLES, 0, triBuffer.components);

    firstPassTarget.disconnect();

    // SECOND PASS - Glow effect

    const quadBuffer = new WebGL.Buffer(gl, "float", 2);
    quadBuffer.set(WebGL.createQuad(-1, -1, 1, 1).arr);

    glowProgram.use();

    glowProgram.addVertexAttribArray("position", quadBuffer);

    const texQuadBuffer = new WebGL.Buffer(gl, "float", 2);
    texQuadBuffer.set(WebGL.createQuad(0, 0, 1, 1).arr);

    glowProgram.addVertexAttribArray("aTexCoord", texQuadBuffer);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, firstPassTarget.texture);

    const uTexLoc = glowProgram.getUniform("uSourceImage");
    uTexLoc.set1i(0);

    const uResolution = glowProgram.getUniform("uResolution");
    uResolution.set2f(canvas.width, canvas.height);

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    console.log(
      quadBuffer.length,
      quadBuffer.components,
      quadBuffer.componentSize
    );
    gl.drawArrays(gl.TRIANGLES, 0, quadBuffer.components);
  }
}

Game.main();
