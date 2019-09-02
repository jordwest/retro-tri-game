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
                uniform vec2 uPosition;
                uniform vec2 uRotation;

                void main() {
                    vec2 scaledPos = position * uScale;
                    vec2 rotatedPos = vec2(
                      scaledPos.x * uRotation.y + scaledPos.y * uRotation.x,
                      scaledPos.y * uRotation.y - scaledPos.x * uRotation.x
                    );
                    gl_Position = vec4(rotatedPos + uPosition, 0.0, 1.0);
                }
            ` as Shader.Source;

      export const frag = `
                precision mediump float;
                uniform vec4 uColor;
                void main() {
                    gl_FragColor = vec4(uColor);
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
                uniform sampler2D uBlurImage;
                varying vec2 texCoord;

                void main() {

                  vec3 srcCol = texture2D(uSourceImage, texCoord).xyz;
                  vec3 glowCol = texture2D(uBlurImage, texCoord).xyz;

                  float scanline = 0.9 + (0.1 * sin(gl_FragCoord.y * 2.0));

                  vec3 outCol = (srcCol + (glowCol * 0.3));
                  outCol = vec3(min(outCol.r, 1.0), min(outCol.g, 1.0), min(outCol.b, 1.0));

                  gl_FragColor = vec4(outCol * scanline, 1.0);
                }
        ` as Shader.Source;
    }

    export namespace Blur {
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
                uniform vec2 uOrientation;
                varying vec2 texCoord;

                void main() {
                  const int kernelHalfWidth = 12;

                  vec3 srcCol = texture2D(uSourceImage, texCoord).xyz;
                  vec3 outCol = vec3(0.0, 0.0, 0.0);
                  vec2 perPixelOffset = (1.0 / uResolution) * uOrientation;
                  for (int x = -kernelHalfWidth; x < kernelHalfWidth; x++) {
                      vec2 offset = perPixelOffset * float(x);
                      float dist = distance(vec2(0.0, 0.0), offset);
                      float magnitude = (1.0 - (dist / float(kernelHalfWidth))) + 0.5;
                      outCol += (texture2D(uSourceImage, texCoord + offset).xyz * magnitude);
                  }
                  outCol = outCol / (float(kernelHalfWidth) * 2.0);
                  gl_FragColor = vec4(outCol * 1.0, 1.0);
                }
        ` as Shader.Source;
    }
  }

  export namespace Vec2 {
    export type T = {
      x: number;
      y: number;
    };

    export const distance = (a: T, b: T) =>
      Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));

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
    export function clear(gl: Ctx) {
      gl.clearColor(0.0, 0.0, 0.0, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }

    namespace GlType {
      export type T = "float" | "int";
      export const toGL = (t: T) => (t === "float" ? GL.FLOAT : GL.INT);
    }

    export class RenderTarget {
      static targets: RenderTarget[] = [];

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
      }

      static create(gl: Ctx, width: number, height: number) {
        return new RenderTarget(gl, width, height);
      }

      private static push(target: RenderTarget) {
        RenderTarget.targets.push(target);
        target.use();
      }

      with(f: () => void) {
        RenderTarget.push(this);
        f();
        RenderTarget.pop();
      }

      private static pop(): RenderTarget | undefined {
        const result = RenderTarget.targets.pop();
        if (result == null) {
          console.error("Attempted to pop RenderTarget when none are bound");
          return;
        }

        const remaining = RenderTarget.targets.length;
        if (remaining >= 1) {
          // Use the previously bound framebuffer
          RenderTarget.targets[remaining - 1].use();
        } else {
          result.disconnect();
        }
        return result;
      }

      private use() {
        this.gl.bindFramebuffer(GL.FRAMEBUFFER, this.framebuffer);
        this.gl.framebufferTexture2D(
          this.gl.FRAMEBUFFER,
          this.gl.COLOR_ATTACHMENT0,
          this.gl.TEXTURE_2D,
          this.texture,
          0
        );
      }

      private disconnect() {
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

      bind() {
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.glBuffer);
      }

      set(arr: BufferSource & { length: number }) {
        this.bind();
        this.gl.bufferData(
          this.gl.ARRAY_BUFFER,
          arr as any,
          this.gl.STATIC_DRAW
        );
        this.length = arr.length;
        this.gl.bindBuffer(GL.ARRAY_BUFFER, null);
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

      set3f(x: number, y: number, z: number) {
        this.gl.uniform3f(this.loc, x, y, z);
      }

      set4f(x: number, y: number, z: number, w: number) {
        this.gl.uniform4f(this.loc, x, y, z, w);
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
        const program = new Program(
          gl,
          Result.unwrap(Shader.Utils.createProgram(gl, vertSource, fragSource))
        );
        program.use();
        return program;
      }

      getAttribLocation(name: string) {
        return this.gl.getAttribLocation(this.glProgram, name);
      }

      addVertexAttribArray(attribName: string, buf: Buffer) {
        this.use();
        buf.bind();
        const location = this.getAttribLocation(attribName);
        this.gl.vertexAttribPointer(
          location,
          buf.componentSize,
          GlType.toGL(buf.glType),
          false,
          0,
          0
        );
        this.gl.enableVertexAttribArray(location);
        this.gl.bindBuffer(GL.ARRAY_BUFFER, null);
        this.gl.useProgram(null);
      }

      getUniform(name: string) {
        this.use();
        const uniformLoc = this.gl.getUniformLocation(this.glProgram, name);
        return new Uniform(this.gl, uniformLoc);
      }
    }
  }

  export namespace Renderers {
    export class Clear {
      gl: WebGL.Ctx;

      constructor(gl: WebGL.Ctx) {
        this.gl = gl;
      }

      render() {
        this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
        this.gl.clear(GL.COLOR_BUFFER_BIT);
      }
    }

    export class Triangle {
      gl: WebGL.Ctx;
      program: WebGL.Program;
      triBuffer: WebGL.Buffer;

      constructor(gl: WebGL.Ctx) {
        this.program = WebGL.Program.create(
          gl,
          Shader.Color.vert,
          Shader.Color.frag
        );

        this.program.use();
        this.triBuffer = new WebGL.Buffer(gl, "float", 2);
        this.triBuffer.set(
          Vec2.FloatArray.from([
            { x: -1, y: -1 },
            { x: 0, y: 1 },
            { x: 1, y: -1 }
          ]).arr
        );

        this.gl = gl;
      }

      render(opts: {
        scale: number;
        position: { x: number; y: number };
        rotation: number;
        color: { r: number; g: number; b: number; a: number };
      }) {
        this.program.use();
        this.program.addVertexAttribArray("position", this.triBuffer);
        const uScale = this.program.getUniform("uScale");
        uScale.set1f(opts.scale);
        const uColor = this.program.getUniform("uColor");
        uColor.set4f(opts.color.r, opts.color.g, opts.color.b, opts.color.a);
        const uPosition = this.program.getUniform("uPosition");
        uPosition.set2f(opts.position.x, opts.position.y);
        const uRotation = this.program.getUniform("uRotation");
        uRotation.set2f(Math.sin(opts.rotation), Math.cos(opts.rotation));
        this.gl.drawArrays(GL.TRIANGLES, 0, this.triBuffer.components);
      }
    }

    export class Glow {
      gl: WebGL.Ctx;
      blurProgram: WebGL.Program;
      glowProgram: WebGL.Program;
      quadBuffer: WebGL.Buffer;
      texQuadBuffer: WebGL.Buffer;
      intermediateTarget: WebGL.RenderTarget;
      intermediateTarget2: WebGL.RenderTarget;

      constructor(gl: WebGL.Ctx) {
        this.gl = gl;
        this.blurProgram = WebGL.Program.create(
          gl,
          Shader.Blur.vert,
          Shader.Blur.frag
        );
        this.glowProgram = WebGL.Program.create(
          gl,
          Shader.Glow.vert,
          Shader.Glow.frag
        );

        this.quadBuffer = new WebGL.Buffer(gl, "float", 2);
        this.texQuadBuffer = new WebGL.Buffer(gl, "float", 2);

        this.quadBuffer.set(WebGL.createQuad(-1, -1, 1, 1).arr);
        this.texQuadBuffer.set(WebGL.createQuad(0, 0, 1, 1).arr);

        this.intermediateTarget = WebGL.RenderTarget.create(
          gl,
          gl.drawingBufferWidth,
          gl.drawingBufferHeight
        );
        this.intermediateTarget2 = WebGL.RenderTarget.create(
          gl,
          gl.drawingBufferWidth,
          gl.drawingBufferHeight
        );
      }

      render(opts: {
        srcTexture: WebGLTexture;
        resolution: { width: number; height: number };
      }) {
        this.blurProgram.use();

        this.blurProgram.addVertexAttribArray("position", this.quadBuffer);
        this.blurProgram.addVertexAttribArray("aTexCoord", this.texQuadBuffer);

        const uTexLoc = this.blurProgram.getUniform("uSourceImage");
        const uResolution = this.blurProgram.getUniform("uResolution");
        const uOrientation = this.blurProgram.getUniform("uOrientation");

        uResolution.set2f(opts.resolution.width, opts.resolution.height);

        this.intermediateTarget.with(() => {
          this.gl.activeTexture(GL.TEXTURE0);
          this.gl.bindTexture(GL.TEXTURE_2D, opts.srcTexture);
          uTexLoc.set1i(0);
          uOrientation.set2f(0.0, 1.0);
          this.gl.drawArrays(GL.TRIANGLES, 0, this.quadBuffer.components);
        });

        this.intermediateTarget2.with(() => {
          this.gl.activeTexture(GL.TEXTURE0);
          this.gl.bindTexture(GL.TEXTURE_2D, this.intermediateTarget.texture);
          uTexLoc.set1i(0);
          uOrientation.set2f(1.0, 0.0);
          this.gl.drawArrays(GL.TRIANGLES, 0, this.quadBuffer.components);
        });

        this.intermediateTarget.with(() => {
          this.gl.activeTexture(GL.TEXTURE0);
          this.gl.bindTexture(GL.TEXTURE_2D, this.intermediateTarget2.texture);
          uTexLoc.set1i(0);
          uOrientation.set2f(0.0, 1.0);
          this.gl.drawArrays(GL.TRIANGLES, 0, this.quadBuffer.components);
        });

        this.intermediateTarget2.with(() => {
          this.gl.activeTexture(GL.TEXTURE0);
          this.gl.bindTexture(GL.TEXTURE_2D, this.intermediateTarget.texture);
          uTexLoc.set1i(0);
          uOrientation.set2f(1.0, 0.0);
          this.gl.drawArrays(GL.TRIANGLES, 0, this.quadBuffer.components);
        });

        this.glowProgram.use();
        this.glowProgram.addVertexAttribArray("position", this.quadBuffer);
        this.glowProgram.addVertexAttribArray("aTexCoord", this.texQuadBuffer);

        const uSrcTexLoc = this.glowProgram.getUniform("uSourceImage");
        const uBlurImage = this.glowProgram.getUniform("uBlurImage");

        this.gl.activeTexture(GL.TEXTURE0);
        this.gl.bindTexture(GL.TEXTURE_2D, opts.srcTexture);
        uSrcTexLoc.set1i(0);
        this.gl.activeTexture(GL.TEXTURE1);
        this.gl.bindTexture(GL.TEXTURE_2D, this.intermediateTarget2.texture);
        uBlurImage.set1i(1);
        this.gl.drawArrays(GL.TRIANGLES, 0, this.quadBuffer.components);
      }
    }
  }

  export namespace State {
    export type EntityId = number & { __entityId: never };
    export type T = {
      nextId: EntityId;
      keys: Map<string, boolean>;
      score: number;
      players: Map<EntityId, Player>;
      firingRate: Map<EntityId, FiringRate.C>;
      positions: Map<EntityId, Vec2.T>;
      enemy: Map<EntityId, boolean>;
      velocities: Map<EntityId, Vec2.T>;
      powerups: Map<EntityId, "trigun" | "health" | "healthupgrade">;
      headings: Map<EntityId, number>;
      damping: Map<EntityId, number>;
      renderables: Map<EntityId, Renderable.C>;
      lifetimes: Map<EntityId, Lifetime.C>;
      dead: Map<EntityId, boolean>;
    };

    export function checkEntityExists<T>(
      id: EntityId,
      desc: string,
      v: T | undefined | null
    ): T {
      if (v == null) {
        console.error(
          "Expected to find entity with id",
          id,
          " for ",
          desc,
          ", got",
          v
        );
        throw new Error("Assertion failed");
      }
      return v;
    }

    export function addPlayer(state: T) {
      const id = getId(state);
      state.renderables.set(id, { type: "player" });
      state.headings.set(id, 0);
      state.firingRate.set(id, { timeRemain: 0, timeTotal: 0.3 });
      state.players.set(id, { health: 5, maxHealth: 5, trigunPowerup: false });
      state.positions.set(id, { x: 0.0, y: -0.8 });
    }

    export function addEnemy(state: T) {
      const id = getId(state);
      state.renderables.set(id, { type: "enemy" });
      state.headings.set(id, Math.PI);
      state.enemy.set(id, true);
      state.firingRate.set(id, {
        timeTotal: 0.8 + Math.random() * 0.4,
        timeRemain: 1.0 + Math.random()
      });
      state.positions.set(id, {
        x: -0.8 + Math.random() * 1.8,
        y: 1.0 + Math.random() * 0.8
      });
    }

    export function addPowerup(
      state: T,
      pos: Vec2.T,
      powerupType: "trigun" | "health" | "healthupgrade"
    ) {
      let id = getId(state);
      state.renderables.set(id, { type: "powerup" });
      state.positions.set(id, { ...pos });
      state.headings.set(id, 0);
      state.powerups.set(id, powerupType);
      state.velocities.set(id, {
        x: 0,
        y: -0.3
      });
    }

    export function addExplosion(state: T, pos: Vec2.T, count: number = 200) {
      for (let i = 0; i < count; i++) {
        addParticle(state, pos);
      }
    }

    export function addBullet(state: T, pos: Vec2.T, heading: number) {
      let id = getId(state);
      state.renderables.set(id, { type: "bullet" });
      state.positions.set(id, { ...pos });
      state.headings.set(id, heading);
      state.velocities.set(id, {
        x: Math.cos(heading + Math.PI / 2) * 1.8,
        y: Math.sin(heading + Math.PI / 2) * 1.8
      });
    }

    export function addParticle(state: T, pos: Vec2.T) {
      let id = getId(state);
      state.renderables.set(id, { type: "particle" });
      state.positions.set(id, { ...pos });
      state.headings.set(id, Math.random() * Math.PI * 2);
      state.lifetimes.set(id, { age: 0, lifespan: 0.6 });
      state.velocities.set(id, {
        x: -1.5 + Math.random() * 3.0,
        y: -1.5 + Math.random() * 3.0
      });
      state.damping.set(id, 1.5 + Math.random() * 3.0);
    }

    export function create(): T {
      const state: T = {
        nextId: 0 as EntityId,
        score: 0,
        keys: new Map(),
        players: new Map(),
        positions: new Map(),
        powerups: new Map(),
        firingRate: new Map(),
        renderables: new Map(),
        enemy: new Map(),
        lifetimes: new Map(),
        headings: new Map(),
        velocities: new Map(),
        damping: new Map(),
        dead: new Map()
      };
      return state;
    }

    export function getId(state: T): EntityId {
      const nextId = state.nextId;
      state.nextId++;
      return nextId;
    }

    export namespace Renderable {
      export type C =
        | { type: "player" }
        | { type: "enemy" }
        | { type: "bullet" }
        | { type: "powerup" }
        | { type: "particle" };
    }

    export namespace Lifetime {
      export type C = { age: number; lifespan: number };
    }

    export namespace FiringRate {
      export type C = { timeRemain: number; timeTotal: number };
    }

    export type Player = {
      health: number;
      maxHealth: number;
      trigunPowerup: boolean;
    };

    export namespace Systems {
      export function tick(state: T, time: number) {
        let player: State.Player, playerId: EntityId;

        state.players.forEach((p, pId) => {
          player = p;
          playerId = pId;
        });

        state.lifetimes.forEach((l, entityId) => {
          l.age += time;
          if (l.age > l.lifespan) {
            state.dead.set(entityId, true);
          }
        });

        state.velocities.forEach((v, entityID) => {
          let position = checkEntityExists(
            entityID,
            "position for velocity",
            state.positions.get(entityID)
          );
          let damping = state.damping.get(entityID);

          position.x += v.x * time;
          position.y += v.y * time;
          if (damping != null) {
            v.x = v.x * (1 - damping * time);
            v.y = v.y * (1 - damping * time);
          }
        });

        state.firingRate.forEach((v, id) => {
          // Don't let it go below zero
          v.timeRemain = Math.max(v.timeRemain - time, 0);

          if (
            v.timeRemain === 0 &&
            state.renderables.get(id).type === "enemy"
          ) {
            const pos = state.positions.get(id);
            if (pos.y <= 0.8) {
              // Check we're on screen before shooting
              addBullet(state, { x: pos.x, y: pos.y - 0.1 }, Math.PI);
              v.timeRemain = v.timeTotal;
            }
          }
        });

        state.enemy.forEach((s, sid) => {
          const enemyPos = state.positions.get(sid);

          if (enemyPos.y > 0.8) {
            // Off screen, slowly move on screen
            enemyPos.y = enemyPos.y - 0.2 * time;
          }
          state.renderables.forEach((r, rid) => {
            const bulletPos = state.positions.get(rid);
            if (r.type === "bullet") {
              const dist = Vec2.distance(enemyPos, bulletPos);
              if (dist < 0.05) {
                state.score += 10;
                document.getElementById(
                  "score"
                ).innerText = state.score.toString();
                state.dead.set(sid, true);
                let trigunChance = 0;
                let healthChance = 0.05;
                let maxHealthChance = 0;
                if (!player.trigunPowerup && state.score > 200) {
                  trigunChance = 0.05;
                }

                if (player.health < 3) {
                  healthChance = 0.08;
                }

                if (player.maxHealth < 10 && state.score > 300) {
                  maxHealthChance = 0.04;
                }

                if (Math.random() < trigunChance) {
                  addPowerup(state, enemyPos, "trigun");
                } else if (Math.random() < maxHealthChance) {
                  addPowerup(state, enemyPos, "healthupgrade");
                } else if (Math.random() < healthChance) {
                  addPowerup(state, enemyPos, "health");
                }
                addExplosion(state, enemyPos);
                state.dead.set(rid, true);
              }
            }
          });
        });

        let targetEnemies = 4;
        if (state.score > 100) {
          targetEnemies = 4 + state.score / 50;
        }

        if (state.enemy.size <= targetEnemies) {
          Game.State.addEnemy(state);
        }

        const moveSpeed = 1.2;
        state.players.forEach((player, playerId) => {
          const playerPos = state.positions.get(playerId);

          state.powerups.forEach((powerupType, powerupId) => {
            const powerupPos = state.positions.get(powerupId);
            if (Vec2.distance(playerPos, powerupPos) < 0.05) {
              state.dead.set(powerupId, true);
              switch (powerupType) {
                case "trigun":
                  player.trigunPowerup = true;
                  break;
                case "health":
                  player.health = Math.min(player.health + 1, player.maxHealth);
                  break;
                case "healthupgrade":
                  player.maxHealth = Math.min(player.maxHealth + 1, 10);
                  player.health = player.maxHealth;
                  break;
              }
            }
          });

          if (
            state.keys.get("ArrowUp") === true ||
            state.keys.get("w") === true
          ) {
            playerPos.y += moveSpeed * time;
          }
          if (
            state.keys.get("ArrowDown") === true ||
            state.keys.get("s") === true
          ) {
            playerPos.y -= moveSpeed * time;
          }
          if (
            state.keys.get("ArrowLeft") === true ||
            state.keys.get("a") === true
          ) {
            playerPos.x -= moveSpeed * time;
          }
          if (
            state.keys.get("ArrowRight") === true ||
            state.keys.get("d") === true
          ) {
            playerPos.x += moveSpeed * time;
          }
          if (state.keys.get("x") === true || state.keys.get("j") === true) {
            const firingRate = state.firingRate.get(playerId);
            if (firingRate && firingRate.timeRemain > 0) {
              // Not ready yet
            } else {
              if (firingRate) {
                firingRate.timeRemain = firingRate.timeTotal;
              }
              const position = state.positions.get(playerId);
              const playerHeading = state.headings.get(playerId);
              if (player.trigunPowerup) {
                addBullet(
                  state,
                  { x: position.x + 0.03, y: position.y + 0.11 },
                  playerHeading - 0.15
                );
                addBullet(
                  state,
                  { x: position.x - 0.03, y: position.y + 0.11 },
                  playerHeading + 0.15
                );
              }
              addBullet(
                state,
                { x: position.x, y: position.y + 0.11 },
                state.headings.get(playerId)
              );
            }
          }

          // Check for any bullets
          state.renderables.forEach((r, bulletId) => {
            const bulletPos = state.positions.get(bulletId);
            if (r.type === "bullet") {
              const dist = Math.sqrt(
                Math.pow(playerPos.x - bulletPos.x, 2) +
                  Math.pow(playerPos.y - bulletPos.y, 2)
              );
              if (dist < 0.05) {
                if (player.health > 1) {
                  player.health = player.health - 1;
                  state.dead.set(bulletId, true);
                  addExplosion(state, playerPos, 10);
                } else {
                  state.dead.set(playerId, true);
                  addExplosion(state, playerPos);
                }
              }
            }
          });
        });
      }

      export function cleanup(state: T) {
        let cleanupIds: EntityId[] = [];
        state.dead.forEach((isDead, id) => {
          if (isDead == true) {
            cleanupIds.push(id);
          }
        });
        cleanupIds.forEach(id => {
          state.renderables.delete(id);
          state.lifetimes.delete(id);
          state.velocities.delete(id);
          state.headings.delete(id);
          state.players.delete(id);
          state.damping.delete(id);
          state.powerups.delete(id);
          state.positions.delete(id);
          state.enemy.delete(id);
          state.firingRate.delete(id);
          state.dead.delete(id);
        });
      }

      export function render(state: T, triRenderer: Renderers.Triangle) {
        state.players.forEach(player => {
          // Draw health bar first
          for (let i = 1; i <= player.maxHealth; i++) {
            const col =
              player.health >= i
                ? { r: 1, g: 0, b: 0, a: 1 }
                : { r: 0.3, g: 0, b: 0, a: 1 };
            triRenderer.render({
              scale: 0.02,
              position: { x: -0.8, y: 0.6 + i * 0.02 },
              rotation: 0,
              color: col
            });
          }
        });
        state.renderables.forEach((renderable, entityId) => {
          const pos = checkEntityExists(
            entityId,
            "position for renderable",
            state.positions.get(entityId)
          );
          const rotation = checkEntityExists(
            entityId,
            "heading for renderable",
            state.headings.get(entityId)
          );
          switch (renderable.type) {
            case "player":
              triRenderer.render({
                scale: 0.03 + Math.random() * 0.01,
                position: { x: pos.x, y: pos.y - 0.1 },
                rotation: rotation + Math.PI,
                color: { r: 1, g: 1, b: 0, a: 1 }
              });
              triRenderer.render({
                scale: 0.08,
                position: pos,
                rotation,
                color: { r: 0, g: 0, b: 1, a: 1 }
              });
              break;
            case "enemy":
              triRenderer.render({
                scale: 0.05,
                position: pos,
                rotation,
                color: { r: 1, g: 0, b: 0, a: 1 }
              });
              break;
            case "bullet":
              triRenderer.render({
                scale: 0.03,
                position: pos,
                rotation,
                color: { r: 1, g: 1, b: 1, a: 1 }
              });
              break;
            case "powerup":
              const powerupType = state.powerups.get(entityId);
              triRenderer.render({
                scale: 0.04,
                position: pos,
                rotation,
                color: { r: 0, g: 0, b: 1, a: 1 }
              });
              switch (powerupType) {
                case "healthupgrade":
                case "health":
                  triRenderer.render({
                    scale: 0.01,
                    position: pos,
                    rotation: rotation + Math.PI,
                    color: { r: 1, g: 0, b: 0, a: 1 }
                  });
                  break;
                case "trigun":
                  triRenderer.render({
                    scale: 0.01,
                    position: pos,
                    rotation: rotation + Math.PI,
                    color: { r: 1, g: 1, b: 1, a: 1 }
                  });
                  break;
              }
              break;
            case "particle":
              const lifetime = checkEntityExists(
                entityId,
                "lifetime for renderable particle",
                state.lifetimes.get(entityId)
              );
              const color = 1.0 - lifetime.age / lifetime.lifespan;
              triRenderer.render({
                scale: 0.02,
                position: pos,
                rotation,
                color: { r: color, g: 0, b: 0, a: color }
              });
              break;
          }
        });
      }
    }
  }

  export function main() {
    const canvas = document.getElementById("game") as HTMLCanvasElement;

    // canvas.width = window.devicePixelRatio * window.innerWidth;
    // canvas.height = window.devicePixelRatio * window.innerHeight;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const gameState = Game.State.create();
    Game.State.addPlayer(gameState);
    const handledKeys = [
      "ArrowUp",
      "ArrowDown",
      "ArrowLeft",
      "ArrowRight",
      "w",
      "a",
      "s",
      "d",
      "j",
      "x"
    ];
    document.addEventListener("keydown", e => {
      if (handledKeys.includes(e.key)) {
        e.preventDefault();
        gameState.keys.set(e.key, true);
      }
    });
    document.addEventListener("keyup", e => {
      if (handledKeys.includes(e.key)) {
        e.preventDefault();
        gameState.keys.set(e.key, false);
      }
    });

    const gl = canvas.getContext("webgl", {
      alpha: false,
      premultipliedAlpha: false
    });
    const triangleRenderer = new Renderers.Triangle(gl);
    const glowRenderer = new Renderers.Glow(gl);

    let lastT = undefined;
    const animationFrame = (t: number) => {
      if (lastT == null) {
        lastT = t;
        requestAnimationFrame(animationFrame);
        return;
      }

      const dt = (t - lastT) / 1000;

      const baseRenderTarget = WebGL.RenderTarget.create(
        gl,
        canvas.width,
        canvas.height
      );

      baseRenderTarget.with(() => {
        gl.blendFunc(GL.SRC_ALPHA, GL.ONE_MINUS_SRC_ALPHA);
        gl.blendFunc(GL.ONE, GL.ONE_MINUS_SRC_ALPHA);
        WebGL.clear(gl);

        Game.State.Systems.cleanup(gameState);
        Game.State.Systems.tick(gameState, dt);
        Game.State.Systems.render(gameState, triangleRenderer);
      });

      WebGL.clear(gl);

      const { width, height } = canvas;
      glowRenderer.render({
        srcTexture: baseRenderTarget.texture,
        resolution: { width, height }
      });

      lastT = t;
      requestAnimationFrame(animationFrame);
    };

    // Start everything
    requestAnimationFrame(animationFrame);
  }
}

Game.main();
console.log("== CONTROLS ==");
console.log("Arrow keys, X to shoot");
console.log("or");
console.log("WASD, J to shoot");
