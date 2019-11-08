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