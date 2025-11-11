//
//  Shaders.metal
//  Slate
//
//  Created by Penny Marshall on 10/27/25.
//

#include <metal_stdlib>
using namespace metal;

struct ViewUniforms {
    float2 anchorLocal;
    float2 anchorNDC;
    float2 tileToNDC;
    float zoomMantissa;
    int   zoomExponent;
    float cosTheta;
    float sinTheta;
    float padding;
};

struct TileUniforms {
    float2 tileDelta;
};

vertex float4 vertex_main(uint vertexID [[vertex_id]],
                         constant float2 *localPositions [[buffer(0)]],
                         constant TileUniforms &tile [[buffer(1)]],
                         constant ViewUniforms &view [[buffer(2)]]) {
    float2 local = localPositions[vertexID];
    float2 delta = tile.tileDelta + (local - view.anchorLocal);
    float2 rotated = float2(delta.x * view.cosTheta + delta.y * view.sinTheta,
                            -delta.x * view.sinTheta + delta.y * view.cosTheta);
    float2 ndcDelta = rotated * view.tileToNDC;
    float scalePow2 = exp2((float)view.zoomExponent);
    float totalScale = scalePow2 * view.zoomMantissa;
    float2 ndc = view.anchorNDC + ndcDelta * totalScale;
    return float4(ndc, 0.0, 1.0);
}

fragment float4 fragment_main() {
    return float4(1.0, 0.0, 0.0, 1.0);
}
