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

vertex float4 vertex_main(uint vertexID [[vertex_id]],
                         constant float2 *positions [[buffer(0)]],
                         constant Transform *transform [[buffer(1)]]) {
    float2 pos = positions[vertexID];
    
    // Vertices are in NDC at identity transform
    // Apply current view transform
    //1. Rotate
    float rotX = pos.x * cos(transform->rotationAngle) + pos.y * sin(transform->rotationAngle); /*x' = x×cos(θ) + y×sin(θ)*/
    float rotY = -pos.x * sin(transform->rotationAngle) + pos.y * cos(transform->rotationAngle);/*y' = -x×sin(θ) + y×cos(θ)*/
    
//2. Zoom
    float zoomedX = rotX * transform->zoomScale;
    float zoomedY = rotY * transform->zoomScale;
    
//3. Pan
    float panX = (transform->panOffset.x / transform->screenWidth) * 2.0;
    float panY = -(transform->panOffset.y / transform->screenHeight) * 2.0;
    
    float2 transformed = float2(zoomedX, zoomedY) + float2(panX, panY);
    
    return float4(transformed, 0.0, 1.0);
}

fragment float4 fragment_main(float4 in [[stage_in]]) {
    return float4(0.0, 1.0, 0.0, 1.0);
}
