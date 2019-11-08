precision mediump float;
attribute vec2 position;
attribute vec2 aTexCoord;
varying vec2 texCoord;

void main() {
    texCoord = aTexCoord;
    gl_Position = vec4(position, 0.0, 1.0);
}