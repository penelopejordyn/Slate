// Shaders.metal houses the Metal shading functions for strokes, cards, and color utilities.
#include <metal_stdlib>
using namespace metal;

/// GPU-Side Rendering Transform (ICB Optimized)
/// Position offset is applied in the vertex shader, not on CPU
struct StrokeTransform {
    float2 relativeOffset;  // Stroke position relative to camera (Floating Origin)
    float2 rotatedOffsetScreen; // Relative offset rotated and scaled to screen pixels
    float zoomScale;        // Current zoom level
    float screenWidth;      // Screen dimensions for NDC conversion
    float screenHeight;
    float rotationAngle;    // Camera rotation
    float halfPixelWidth;   // Half-width of stroke in screen pixels (for screen-space extrusion)
    float featherPx;        // Feather amount in pixels for SDF edge
    float depth;            // Metal NDC depth [0, 1] (smaller = closer)
};

/// GPU segment instance data for SDF strokes
struct SegmentInstance {
    float2 p0;      // stroke-local world
    float2 p1;      // stroke-local world
    float4 color;
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

// MARK: - SDF Segment Rendering

struct QuadIn {
    float2 corner [[attribute(0)]]; // (0..1, 0..1)
};

struct SegmentOut {
    float4 position [[position]];
    float2 fragScreen;    // pixel coords of this fragment
    float2 p0Screen;      // segment endpoint in pixels
    float2 p1Screen;
    float4 color;
};

vertex SegmentOut vertex_segment_sdf(
    QuadIn vin [[stage_in]],
    constant StrokeTransform *t [[buffer(1)]],
    const device SegmentInstance *instances [[buffer(2)]],
    uint iid [[instance_id]]
) {
    SegmentInstance seg = instances[iid];

    // 1) Rotate local points
    float c = cos(t->rotationAngle);
    float s = sin(t->rotationAngle);

    float2 r0 = float2(seg.p0.x * c - seg.p0.y * s, seg.p0.x * s + seg.p0.y * c);
    float2 r1 = float2(seg.p1.x * c - seg.p1.y * s, seg.p1.x * s + seg.p1.y * c);

    // 2) Convert to screen pixel space using pre-rotated stroke offset
    float2 p0 = r0 * t->zoomScale + t->rotatedOffsetScreen;
    float2 p1 = r1 * t->zoomScale + t->rotatedOffsetScreen;

    float2 d = p1 - p0;
    float len = length(d);
    float2 dir = (len > 0.0) ? (d / len) : float2(1.0, 0.0);
    float2 nrm = float2(-dir.y, dir.x);

    float R = t->halfPixelWidth;

    // 4) Build rectangle around capsule in screen space
    // x in [-R, len+R], y in [-R, +R]
    float x = mix(-R, len + R, vin.corner.x);
    float y = mix(-R, +R,        vin.corner.y);

    float2 screenPos = p0 + dir * x + nrm * y;

    // 5) Convert screen pixels to NDC
    float ndcX = (screenPos.x / t->screenWidth) * 2.0;
    float ndcY = -(screenPos.y / t->screenHeight) * 2.0;

    SegmentOut out;
    out.position = float4(ndcX, ndcY, t->depth, 1.0);
    out.fragScreen = screenPos;
    out.p0Screen = p0;
    out.p1Screen = p1;
    out.color = seg.color;
    return out;
}

fragment float4 fragment_segment_sdf(
    SegmentOut in [[stage_in]],
    constant StrokeTransform *t [[buffer(1)]]
) {
    float2 p = in.fragScreen;
    float2 a = in.p0Screen;
    float2 b = in.p1Screen;

    float2 ab = b - a;
    float denom = dot(ab, ab);

    float h = 0.0;
    if (denom > 0.0) {
        h = dot(p - a, ab) / denom;
        h = clamp(h, 0.0, 1.0);
    }

    float2 closest = a + h * ab;
    float dist = length(p - closest);

    float R = t->halfPixelWidth;
    float f = max(t->featherPx, 0.5);

    float alpha = 1.0 - smoothstep(R - f, R + f, dist);

    if (alpha <= 0.0) {
        discard_fragment();
    }

    float4 color = in.color;
    color.a *= alpha;
    return color;
}

vertex VertexOut vertex_main(VertexIn in [[stage_in]],
                             constant StrokeTransform *transform [[buffer(1)]]) {
    // SCREEN-SPACE EXTRUSION APPROACH
    // Key innovation: Thickness is constant in pixels, not world units
    // This prevents strokes from becoming blobs at high zoom

    // 1. Centerline in camera-relative world space
    float2 worldRelative = in.position + transform->relativeOffset;

    // 2. Rotate around camera center
    float c = cos(transform->rotationAngle);
    float s = sin(transform->rotationAngle);
    float2 rot = float2(
        worldRelative.x * c - worldRelative.y * s,
        worldRelative.x * s + worldRelative.y * c
    );

    // 3. Project centerline to NDC (no width yet)
    float2 zoomed = rot * transform->zoomScale;
    float2 ndcCenter = float2(
        (zoomed.x / transform->screenWidth) * 2.0,
        -(zoomed.y / transform->screenHeight) * 2.0
    );

    // 4. Calculate screen-space normal for extrusion
    // We use a simple approximation: rotate the "up" vector by the camera rotation
    // This gives us a consistent perpendicular direction in screen space
    float2 localUp = float2(0.0, 1.0);
    float2 rotUp = float2(
        localUp.x * c - localUp.y * s,
        localUp.x * s + localUp.y * c
    );
    float2 zoomedUp = rotUp * transform->zoomScale;
    float2 ndcUp = float2(
        (zoomedUp.x / transform->screenWidth) * 2.0,
        -(zoomedUp.y / transform->screenHeight) * 2.0
    );

    // Perpendicular to "up" gives us the extrusion direction
    float2 ndcNormal = normalize(float2(-ndcUp.y, ndcUp.x));

    // 5. Convert pixel radius to NDC units
    float pixelToNDC = 2.0 / transform->screenHeight;
    float halfWidthNDC = transform->halfPixelWidth * pixelToNDC;

    // 6. Extrude using side flag from uv.y (-1 or +1)
    float side = in.uv.y;
    float2 ndcPos = ndcCenter + ndcNormal * (side * halfWidthNDC);

    VertexOut out;
    out.position = float4(ndcPos, transform->depth, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    // SQUARE BRUSH WITH ROUND CAPS
    // Local stroke coordinates:
    // t: along stroke (0 at start, 1 at end)
    // s: across stroke (-1 at one edge, +1 at the other)
    float t = clamp(in.uv.x, 0.0, 1.0);
    float s = in.uv.y;

    // --- 1. Handle side edges (square brush) ---

    // We want a hard edge at |s| = 1, maybe with a tiny feather.
    float distSide = fabs(s); // 0 at center, 1 at edge

    // Feather width in "s" space
    float sideFeather = 0.02; // tune (0 = completely hard edge)
    float alphaSide = 1.0 - smoothstep(1.0 - sideFeather, 1.0, distSide);
    // Inside |s|<1-sideFeather -> alphaSide ~ 1
    // Outside |s|>1 -> alphaSide ~ 0

    // --- 2. Handle round caps (start & end) ---

    // How much of the stroke length is used for rounded caps (normalized):
    // e.g., 0.5 would mean caps cover 50% each = almost a circle for short strokes.
    // Smaller = shorter caps.
    float capFrac = 0.2;  // 20% of normalized length at each end, tune this

    float alphaCap = 1.0;

    // Start cap (t near 0)
    if (t < capFrac) {
        // Map t from [0, capFrac] -> [-1, 0]; center of cap at (-0.5, 0)
        float x = (t / capFrac) - 1.0;  // -1 at start, 0 at cap boundary
        float2 p = float2(x, s);       // local cap coords
        float d = length(p);           // distance from cap center

        // Radius = 1.0 in this local space (covers full width)
        float capFeather = 0.02; // edge softness
        alphaCap = 1.0 - smoothstep(1.0 - capFeather, 1.0, d);
    }
    // End cap (t near 1)
    else if (t > 1.0 - capFrac) {
        // Map t from [1-capFrac, 1] -> [0, 1]; center of cap at (0, 0) in local space
        float u = (t - (1.0 - capFrac)) / capFrac; // 0 at junction, 1 at far end
        float2 p = float2(u, s);                    // local cap coords
        float d = length(p);                        // distance from cap center at origin

        // Radius = 1.0 in this local space (covers full width)
        float capFeather = 0.02;
        alphaCap = 1.0 - smoothstep(1.0 - capFeather, 1.0, d);
    }

    // Combine side + caps:
    // - In the middle section alphaCap stays ~1; near ends it's shaped by circle.
    float alpha = alphaSide * alphaCap;

    // Clamp for safety
    alpha = clamp(alpha, 0.0, 1.0);

    // Apply alpha to color
    float4 color = in.color;
    color.a *= alpha;

    return color;
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
