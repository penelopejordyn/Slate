//
//  Shaders.metal
//  Slate
//
//  Created by Penny Marshall on 10/27/25.
//

#include <metal_stdlib>
using namespace metal;

struct Transform {
    float2 panOffset;
    float zoomScale;
    float screenWidth;
    float screenHeight;
    float rotationAngle;
};

struct StrokeVertex {
    float2 high;
    float2 low;
};

vertex float4 vertex_main(uint vertexID [[vertex_id]],
                         constant StrokeVertex *positions [[buffer(0)]],
                         constant Transform *transform [[buffer(1)]]) {
    StrokeVertex vert = positions[vertexID];
    float2 world = vert.high + vert.low;

    // Convert world pixels → model NDC (identity view transform)
    float modelX = (world.x / transform->screenWidth) * 2.0 - 1.0;
    float modelY = -((world.y / transform->screenHeight) * 2.0 - 1.0);

    // Rotate
    float c = cos(transform->rotationAngle);
    float s = sin(transform->rotationAngle);
    float rotX = modelX * c + modelY * s;
    float rotY = -modelX * s + modelY * c;

    // Zoom
    float zoomedX = rotX * transform->zoomScale;
    float zoomedY = rotY * transform->zoomScale;

    // Pan (provided in pixels)
    float panX = (transform->panOffset.x / transform->screenWidth) * 2.0;
    float panY = -(transform->panOffset.y / transform->screenHeight) * 2.0;

    float2 transformed = float2(zoomedX + panX, zoomedY + panY);

    return float4(transformed, 0.0, 1.0);
}

fragment float4 fragment_main(float4 in [[stage_in]]) {
    return float4(0.0, 1.0, 0.0, 1.0);
}
