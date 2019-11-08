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