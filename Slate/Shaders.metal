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
};

vertex float4 vertex_main(uint vertexID [[vertex_id]],
                         constant float2 *positions [[buffer(0)]],
                         constant Transform *transform [[buffer(1)]]) {
    float2 pos = positions[vertexID];
    
    // 1. Apply zoom around center
    float2 zoomed = pos * transform->zoomScale;
    
    // 2. Apply pan, but COMPENSATE for zoom
    //    Divide pan by zoom so it stays constant in screen space
    float panNormX = (transform->panOffset.x / transform->screenWidth) * 2.0 / transform->zoomScale;
    float panNormY = (transform->panOffset.y / transform->screenHeight) * 2.0 / transform->zoomScale;
    
    float2 transformed = zoomed + float2(panNormX, -panNormY);
    
    return float4(transformed, 0.0, 1.0);
}

fragment float4 fragment_main(float4 in [[stage_in]]) {
    return float4(0.0, 1.0, 0.0, 1.0);
}
