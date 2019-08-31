type GL = WebGLRenderingContext;
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
        gl: GL,
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
      export const frag = `
                void main() {
                    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
                }
            ` as Shader.Source;

      export const vert = `
                attribute vec2 position;

                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
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

  export function main() {
    console.log("hello");
    const canvas = document.getElementById("game") as HTMLCanvasElement;

    const gl = canvas.getContext("webgl");
    const program = Result.unwrap(
      Shader.Utils.createProgram(gl, Shader.Color.vert, Shader.Color.frag)
    );
    console.log(program);

    const buffer = gl.createBuffer();
    const data = Vec2.FloatArray.from([
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 }
    ]);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data.arr, gl.STATIC_DRAW);

    const attribLocation = gl.getAttribLocation(program, "position");
    gl.vertexAttribPointer(attribLocation, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(attribLocation);

    gl.useProgram(program);
    gl.drawArrays(gl.TRIANGLES, 0, data.length);
  }
}

// const audioCtx = new AudioContext();
// const ws = audioCtx.createWaveShaper();

Game.main();
