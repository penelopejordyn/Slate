// Shaders.metal houses the Metal shading functions for strokes, cards, and color utilities.
#include <metal_stdlib>
using namespace metal;

/// GPU-Side Rendering Transform (ICB Optimized)
/// Position offset is applied in the vertex shader, not on CPU
struct StrokeTransform {
    float2 relativeOffset;  // Stroke position relative to camera (Floating Origin)
    float zoomScale;        // Current zoom level
    float screenWidth;      // Screen dimensions for NDC conversion
    float screenHeight;
    float rotationAngle;    // Camera rotation
    float halfPixelWidth;   // Half of desired stroke width in pixels
    uint vertexCount;       // Total vertices in the strip (for safe neighbor lookup)
};

/// Vertex input for batched stroke rendering
struct VertexIn {
    float2 position [[attribute(0)]];  // Camera-relative position (calculated on CPU)
    float2 uv       [[attribute(1)]];  // Texture coordinate
    float4 color    [[attribute(2)]];  // Vertex color (baked for batching)
};

/// Vertex output passed to fragment shader
struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

vertex VertexOut vertex_main(const device VertexIn *vertices [[buffer(0)]],
                             uint vid [[vertex_id]],
                             constant StrokeTransform *transform [[buffer(1)]]) {
    VertexIn in = vertices[vid];

    // Helper to transform a centerline position into screen space (pixels)
    auto toScreen = ^(float2 position) {
        float2 worldRelative = position + transform->relativeOffset;
        float c = cos(transform->rotationAngle);
        float s = sin(transform->rotationAngle);
        float rotX = worldRelative.x * c - worldRelative.y * s;
        float rotY = worldRelative.x * s + worldRelative.y * c;
        return float2(rotX, rotY) * transform->zoomScale;
    };

    uint sampleIndex = vid / 2;              // Two vertices per center sample
    uint baseIndex = sampleIndex * 2;        // Even index stores the center position we need
    uint totalSamples = max<uint>(1, transform->vertexCount / 2);
    uint prevSample = sampleIndex > 0 ? sampleIndex - 1 : 0;
    uint nextSample = min(sampleIndex + 1, totalSamples - 1);

    float2 prevPos = toScreen(vertices[prevSample * 2].position);
    float2 currPos = toScreen(vertices[baseIndex].position);
    float2 nextPos = toScreen(vertices[nextSample * 2].position);

    float2 tangent = nextPos - prevPos;
    float len = length(tangent);
    float2 normal = len > 1e-5 ? float2(-tangent.y, tangent.x) / len : float2(0.0, 1.0);

    // Extrude in screen space before projecting to NDC
    float2 extruded = currPos + normal * (transform->halfPixelWidth * in.uv.y);

    float ndcX = (extruded.x / transform->screenWidth) * 2.0;
    float ndcY = -(extruded.y / transform->screenHeight) * 2.0;

    VertexOut out;
    out.position = float4(ndcX, ndcY, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return in.color;
}

// MARK: - Card Rendering Shaders

/// Transform for cards (not batched, so needs relativeOffset)
struct CardTransform {
    float2 relativeOffset;  // Card position relative to camera
    float zoomScale;
    float screenWidth;
    float screenHeight;
    float rotationAngle;
};

/// Vertex input structure for cards - matches Swift's StrokeVertex
struct CardVertexIn {
    float2 position [[attribute(0)]];  // Local position
    float2 uv       [[attribute(1)]];  // Texture coordinate
};

/// Vertex output structure - passed to fragment shader
struct CardVertexOut {
    float4 position [[position]];  // Clip-space position
    float2 uv;                     // Texture coordinate (interpolated)
};

/// Vertex shader for cards - applies floating origin transform
vertex CardVertexOut vertex_card(CardVertexIn in [[stage_in]],
                                 constant CardTransform *transform [[buffer(1)]]) {
    // Same transform logic as strokes
    float2 worldRelative = in.position + transform->relativeOffset;

    // Rotation
    float c = cos(transform->rotationAngle);
    float s = sin(transform->rotationAngle);
    float rotX = worldRelative.x * c - worldRelative.y * s;
    float rotY = worldRelative.x * s + worldRelative.y * c;

    // Zoom
    float2 zoomed = float2(rotX, rotY) * transform->zoomScale;

    // Projection to NDC
    float ndcX = (zoomed.x / transform->screenWidth) * 2.0;
    float ndcY = -(zoomed.y / transform->screenHeight) * 2.0;

    CardVertexOut out;
    out.position = float4(ndcX, ndcY, 0.0, 1.0);
    out.uv = in.uv;

    return out;
}

/// Fragment shader for solid color cards - no texture required
/// This avoids Metal validation issues when no texture is bound
fragment float4 fragment_card_solid(CardVertexOut in [[stage_in]],
                                    constant float4 &color [[buffer(0)]]) {
    return color;
}

/// Fragment shader for textured cards (images, PDFs)
/// This shader requires a texture to be bound at index 0
fragment float4 fragment_card_texture(CardVertexOut in [[stage_in]],
                                      texture2d<float> cardTexture [[texture(0)]],
                                      sampler cardSampler [[sampler(0)]]) {
    return cardTexture.sample(cardSampler, in.uv);
}

// MARK: - Procedural Background Shaders

/// Shader uniforms for procedural backgrounds
struct CardShaderUniforms {
    float spacing;
    float lineWidth;
    float4 color;
    float cardWidth;
    float cardHeight;
};

/// LINED PAPER SHADER
/// Renders horizontal lines at regular intervals
/// Lines stay sharp at infinite zoom because they're calculated procedurally per-pixel
fragment float4 fragment_card_lined(CardVertexOut in [[stage_in]],
                                    constant float4 &bgColor [[buffer(0)]],
                                    constant CardShaderUniforms &uniforms [[buffer(1)]]) {
    // 1. Calculate position in card units (world units)
    // UV is 0..1. Multiply by height to get local Y position in world units.
    float yPos = in.uv.y * uniforms.cardHeight;

    // 2. Modulo arithmetic to find distance from nearest line
    // spacing is in World Units (e.g. 0.025 at high zoom)
    float dist = fmod(yPos, uniforms.spacing);

    // 3. Anti-aliasing using automatic pixel derivatives
    // fwidth() gives us the rate of change of yPos per pixel
    // This automatically adapts to any zoom level!
    float delta = fwidth(yPos);

    // 4. Calculate distance to center of line
    // Handle wrap-around (distance to 0 or spacing)
    float halfLine = uniforms.lineWidth * 0.5;
    float d = min(dist, uniforms.spacing - dist);

    // 5. AA using pixel derivatives for crisp lines at any zoom
    float alpha = 1.0 - smoothstep(halfLine - delta, halfLine + delta, d);

    // Mix background color and line color
    return mix(bgColor, uniforms.color, alpha);
}

/// GRID PAPER SHADER
/// Renders both horizontal and vertical lines at regular intervals
/// Creates a graph paper effect that stays sharp at infinite zoom
fragment float4 fragment_card_grid(CardVertexOut in [[stage_in]],
                                   constant float4 &bgColor [[buffer(0)]],
                                   constant CardShaderUniforms &uniforms [[buffer(1)]]) {
    // Calculate X and Y positions in card units (world units)
    float xPos = in.uv.x * uniforms.cardWidth;
    float yPos = in.uv.y * uniforms.cardHeight;

    // Calculate distance to nearest grid line in both directions
    float distX = fmod(xPos, uniforms.spacing);
    float distY = fmod(yPos, uniforms.spacing);

    // Anti-aliasing using automatic pixel derivatives
    float deltaX = fwidth(xPos);
    float deltaY = fwidth(yPos);

    // Calculate distance to center of lines (handle wrap-around)
    float halfLine = uniforms.lineWidth * 0.5;
    float dX = min(distX, uniforms.spacing - distX);
    float dY = min(distY, uniforms.spacing - distY);

    // AA using pixel derivatives for crisp lines at any zoom
    float alphaX = 1.0 - smoothstep(halfLine - deltaX, halfLine + deltaX, dX);
    float alphaY = 1.0 - smoothstep(halfLine - deltaY, halfLine + deltaY, dY);

    // Combine horizontal and vertical lines
    float alpha = max(alphaX, alphaY);

    // Mix background color and line color
    return mix(bgColor, uniforms.color, alpha);
}
