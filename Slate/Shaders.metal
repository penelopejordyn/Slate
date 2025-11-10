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
    float2 cameraCenterModel;
};

vertex float4 vertex_main(uint vertexID [[vertex_id]],
                         constant float2 *positions [[buffer(0)]],
                         constant Transform *transform [[buffer(1)]]) {
    float2 offset = positions[vertexID];
    float2 center = transform->cameraCenterModel;

    float c = cos(transform->rotationAngle);
    float s = sin(transform->rotationAngle);

    float2 rotOffset = float2(offset.x * c + offset.y * s,
                              -offset.x * s + offset.y * c);
    float2 rotCenter = float2(center.x * c + center.y * s,
                              -center.x * s + center.y * c);

    float2 zoomed = (rotOffset + rotCenter) * transform->zoomScale;

    float panX = (transform->panOffset.x / transform->screenWidth) * 2.0;
    float panY = -(transform->panOffset.y / transform->screenHeight) * 2.0;

    float2 transformed = zoomed + float2(panX, panY);

    return float4(transformed, 0.0, 1.0);
}

fragment float4 fragment_main(float4 in [[stage_in]]) {
    return float4(0.0, 1.0, 0.0, 1.0);
}
