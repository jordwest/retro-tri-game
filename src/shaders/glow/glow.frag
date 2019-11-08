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