//
//  Shaders.metal
//  Slate
//
//  Created by Penny Marshall on 10/27/25.
//

#include <metal_stdlib>
using namespace metal;

/// Floating Origin Transform
/// The GPU receives ONLY small relative coordinates - never large world coordinates.
/// This eliminates float32 precision issues at extreme zoom levels.
struct StrokeTransform {
    float2 relativeOffset;  // Stroke origin - Camera center (world units, but small!)
    float zoomScale;        // Current zoom level
    float screenWidth;      // Screen dimensions for NDC conversion
    float screenHeight;
    float rotationAngle;    // Camera rotation
};

vertex float4 vertex_main(uint vertexID [[vertex_id]],
                         constant float2 *localPositions [[buffer(0)]],
                         constant StrokeTransform *transform [[buffer(1)]]) {
    // Step A: Position - Add relative offset to local vertex
    // localPosition is small (e.g., 0-100 world units from stroke origin)
    // relativeOffset is small (e.g., ±500 world units from camera center)
    // Result: Small number ± small number = small number (no precision loss!)
    float2 worldRelative = localPositions[vertexID] + transform->relativeOffset;

    // Step B: Rotation - Rotate around (0,0) which is now the camera center
    // 🟢 FIX: Use Standard Clockwise Rotation Matrix
    // x' = x*cos - y*sin
    // y' = x*sin + y*cos
    float c = cos(transform->rotationAngle);
    float s = sin(transform->rotationAngle);
    float rotX = worldRelative.x * c - worldRelative.y * s;
    float rotY = worldRelative.x * s + worldRelative.y * c;

    // Step C: Zoom - Scale by zoom factor
    float2 zoomed = float2(rotX, rotY) * transform->zoomScale;

    // Step D: Projection - Convert to NDC [-1, 1]
    float ndcX = (zoomed.x / transform->screenWidth) * 2.0;
    float ndcY = -(zoomed.y / transform->screenHeight) * 2.0;

    return float4(ndcX, ndcY, 0.0, 1.0);
}

fragment float4 fragment_main(float4 in [[stage_in]]) {
    return float4(0.0, 1.0, 0.0, 1.0);
}

// MARK: - Card Rendering Shaders

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
                                 constant StrokeTransform *transform [[buffer(1)]]) {
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
