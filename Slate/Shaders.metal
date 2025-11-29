// Shaders.metal houses the Metal shading functions for strokes, cards, and color utilities.
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
    //  FIX: Use Standard Clockwise Rotation Matrix
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

// MARK: - Procedural Card Background Shaders

/// Uniforms for procedural card backgrounds (lined paper, grids)
/// Must match the Swift struct exactly for memory layout
struct CardBackgroundUniforms {
    float4 color;           // Base paper color (RGBA)
    float4 lineColor;       // Line color (RGBA)
    float2 cardSize;        // Card dimensions in world units
    float lineWidth;        // Line thickness in pixels
    float spacing;          // Distance between lines in pixels
    float marginTop;        // Top margin (0.0-1.0 percentage)
    float marginBottom;     // Bottom margin (0.0-1.0 percentage)
    float marginLeft;       // Left margin (0.0-1.0 percentage)
    float marginRight;      // Right margin (0.0-1.0 percentage)
    int style;              // 0=Solid, 1=Lined, 2=Grid
    float imageOpacity;     // Image opacity (0.0-1.0)
};

/// Procedural fragment shader for lined and grid backgrounds
/// Draws lines mathematically for infinite resolution and zero memory cost
fragment float4 fragment_card_procedural(CardVertexOut in [[stage_in]],
                                         constant CardBackgroundUniforms &settings [[buffer(0)]]) {
    // 1. Start with base paper color
    float4 finalColor = settings.color;

    // 2. Calculate margin boundaries (UVs go from 0.0 to 1.0)
    bool inMarginTop    = in.uv.y < settings.marginTop;
    bool inMarginBottom = in.uv.y > (1.0 - settings.marginBottom);
    bool inMarginLeft   = in.uv.x < settings.marginLeft;
    bool inMarginRight  = in.uv.x > (1.0 - settings.marginRight);

    bool inSafeZone = !inMarginTop && !inMarginBottom && !inMarginLeft && !inMarginRight;

    // 3. Draw lines (only in safe zone)
    if (settings.style == 1 || settings.style == 2) { // Lined or Grid
        if (inSafeZone) {
            // Convert UV to pixel coordinates for consistent spacing
            float2 posPixels = in.uv * settings.cardSize;

            float lineAlpha = 0.0;
            float halfWidth = settings.lineWidth * 0.5;

            // Horizontal lines (for both lined and grid)
            float distY = fmod(posPixels.y, settings.spacing);
            float lineDistY = min(distY, settings.spacing - distY);
            // Smoothstep for anti-aliased edges
            float alphaY = 1.0 - smoothstep(halfWidth - 0.5, halfWidth + 0.5, lineDistY);
            lineAlpha = max(lineAlpha, alphaY);

            // Vertical lines (grid only)
            if (settings.style == 2) {
                float distX = fmod(posPixels.x, settings.spacing);
                float lineDistX = min(distX, settings.spacing - distX);
                float alphaX = 1.0 - smoothstep(halfWidth - 0.5, halfWidth + 0.5, lineDistX);
                lineAlpha = max(lineAlpha, alphaX);
            }

            // Blend line color with paper color
            finalColor = mix(finalColor, settings.lineColor, lineAlpha * settings.lineColor.a);
        }
    }

    // Draw margin lines for all enabled margins
    // Use the same color and thickness as the regular lines
    float halfMarginWidth = settings.lineWidth * 0.5;
    float marginThicknessUV = settings.lineWidth / settings.cardSize.x; // Convert to UV space

    // Left margin line
    if (settings.marginLeft > 0.0 && abs(in.uv.x - settings.marginLeft) < marginThicknessUV) {
        float alpha = 1.0 - smoothstep(halfMarginWidth - 0.5, halfMarginWidth + 0.5,
                                       abs(in.uv.x - settings.marginLeft) * settings.cardSize.x);
        finalColor = mix(finalColor, settings.lineColor, alpha * settings.lineColor.a);
    }

    // Right margin line
    if (settings.marginRight > 0.0 && abs(in.uv.x - (1.0 - settings.marginRight)) < marginThicknessUV) {
        float alpha = 1.0 - smoothstep(halfMarginWidth - 0.5, halfMarginWidth + 0.5,
                                       abs(in.uv.x - (1.0 - settings.marginRight)) * settings.cardSize.x);
        finalColor = mix(finalColor, settings.lineColor, alpha * settings.lineColor.a);
    }

    // Top margin line
    if (settings.marginTop > 0.0 && abs(in.uv.y - settings.marginTop) < marginThicknessUV) {
        float alpha = 1.0 - smoothstep(halfMarginWidth - 0.5, halfMarginWidth + 0.5,
                                       abs(in.uv.y - settings.marginTop) * settings.cardSize.y);
        finalColor = mix(finalColor, settings.lineColor, alpha * settings.lineColor.a);
    }

    // Bottom margin line
    if (settings.marginBottom > 0.0 && abs(in.uv.y - (1.0 - settings.marginBottom)) < marginThicknessUV) {
        float alpha = 1.0 - smoothstep(halfMarginWidth - 0.5, halfMarginWidth + 0.5,
                                       abs(in.uv.y - (1.0 - settings.marginBottom)) * settings.cardSize.y);
        finalColor = mix(finalColor, settings.lineColor, alpha * settings.lineColor.a);
    }

    return finalColor;
}
