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

struct StrokeVertexIn {
    float2 localPosition;
    uint strokeIndex;
    uint padding;
};

vertex float4 vertex_main(uint vertexID [[vertex_id]],
                         constant StrokeVertexIn *vertices [[buffer(0)]],
                         constant float2 *origins [[buffer(1)]],
                         constant Transform *transform [[buffer(2)]]) {
    StrokeVertexIn v = vertices[vertexID];
    float2 originWorld = origins[v.strokeIndex];

    float modelOriginX = (originWorld.x / transform->screenWidth) * 2.0 - 1.0;
    float modelOriginY = -((originWorld.y / transform->screenHeight) * 2.0 - 1.0);

    float modelOffsetX = (v.localPosition.x / transform->screenWidth) * 2.0;
    float modelOffsetY = -((v.localPosition.y / transform->screenHeight) * 2.0);

    float cosTheta = cos(transform->rotationAngle);
    float sinTheta = sin(transform->rotationAngle);

    float rotatedOriginX = modelOriginX * cosTheta + modelOriginY * sinTheta;
    float rotatedOriginY = -modelOriginX * sinTheta + modelOriginY * cosTheta;

    float rotatedOffsetX = modelOffsetX * cosTheta + modelOffsetY * sinTheta;
    float rotatedOffsetY = -modelOffsetX * sinTheta + modelOffsetY * cosTheta;

    float zoomedOriginX = rotatedOriginX * transform->zoomScale;
    float zoomedOriginY = rotatedOriginY * transform->zoomScale;

    float zoomedOffsetX = rotatedOffsetX * transform->zoomScale;
    float zoomedOffsetY = rotatedOffsetY * transform->zoomScale;

    float panX = (transform->panOffset.x / transform->screenWidth) * 2.0;
    float panY = -(transform->panOffset.y / transform->screenHeight) * 2.0;

    float2 ndc = float2(zoomedOriginX + zoomedOffsetX + panX,
                        zoomedOriginY + zoomedOffsetY + panY);

    return float4(ndc, 0.0, 1.0);
}

fragment float4 fragment_main(float4 in [[stage_in]]) {
    return float4(0.0, 1.0, 0.0, 1.0);
}
