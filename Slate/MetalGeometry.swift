// MetalGeometry.swift contains math utilities for stroke tessellation, coordinate
// conversions, and geometry helpers used by Metal rendering.
import SwiftUI
import Metal
import MetalKit

// MARK: - Vertex Structures

/// Vertex structure with position, UV coordinates, and color for batched rendering
struct StrokeVertex {
    var position: SIMD2<Float>  // Position in local space
    var uv: SIMD2<Float>         // Texture coordinate (U = along stroke, V = across width)
    var color: SIMD4<Float>      // Vertex color (baked for batching)
}

/// Instance data for distance-field stroke segments
struct StrokeSegmentInstance {
    var p0: SIMD2<Float>      // Stroke-local centerline point 0
    var p1: SIMD2<Float>      // Stroke-local centerline point 1
    var color: SIMD4<Float>   // RGBA
}

/// Vertex for the reusable unit quad used by instanced segment rendering
struct QuadVertex {
    var corner: SIMD2<Float>
}

// MARK: - Transform Structures

/// Transform for ICB stroke rendering (position offset calculated on GPU)
struct StrokeTransform {
    var relativeOffset: SIMD2<Float>  // Stroke position relative to camera
    var rotatedOffsetScreen: SIMD2<Float> // Relative offset rotated and scaled to screen pixels
    var zoomScale: Float
    var screenWidth: Float
    var screenHeight: Float
    var rotationAngle: Float
    var halfPixelWidth: Float           // Half-width of stroke in screen pixels (for screen-space extrusion)
    var featherPx: Float                // Feather amount in pixels for SDF edge
    var depth: Float                    // Depth in Metal NDC [0, 1] (smaller = closer)
}

/// Transform for card rendering (not batched, includes offset)
struct CardTransform {
    var relativeOffset: SIMD2<Float>
    var zoomScale: Float
    var screenWidth: Float
    var screenHeight: Float
    var rotationAngle: Float
    var depth: Float
}

struct CardStyleUniforms {
    var cardHalfSize: SIMD2<Float>
    var zoomScale: Float
    var cornerRadiusPx: Float
    var shadowBlurPx: Float
    var shadowOpacity: Float
    var cardOpacity: Float  // Overall card opacity (0.0 - 1.0)
}

// MARK: - Geometry / Tessellation

/// Convert a world (canvas pixel) point to NDC, applying pan/zoom.
///
/// CRITICAL: Uses Double-precision arithmetic throughout to maintain precision
/// for large world coordinates. Only converts to Float at the final NDC step.
/// This prevents precision loss when working with coordinates like 5,000,000.3
func worldPixelToNDC(point w: CGPoint,
                     viewSize: CGSize,
                     panOffset: SIMD2<Float>,
                     zoomScale: Float) -> SIMD2<Float> {
    // Keep everything as Double for precision!
    let cx = viewSize.width * 0.5
    let cy = viewSize.height * 0.5

    // w.x and w.y are already Double (CGFloat is Double on 64-bit)
    let wx = w.x
    let wy = w.y

    // Remove center (world -> centered) - still Double
    let centeredX = wx - cx
    let centeredY = wy - cy

    // Apply zoom (centered -> zoomed) - still Double
    let zx = centeredX * Double(zoomScale)
    let zy = centeredY * Double(zoomScale)

    // Apply pan (in pixels) - still Double
    let px = zx + Double(panOffset.x)
    let py = zy + Double(panOffset.y)

    // Back to screen pixels - still Double
    let sx = px + cx
    let sy = py + cy

    // Screen pixels -> NDC - ONLY NOW convert to Float
    let ndcX = Float((sx / viewSize.width) * 2.0 - 1.0)
    let ndcY = Float(-((sy / viewSize.height) * 2.0 - 1.0))

    return SIMD2<Float>(ndcX, ndcY)
}

/// Create triangles for a stroke from world (canvas pixel) center points.
func tessellateStroke(centerPoints: [CGPoint],
                      width: CGFloat,
                      viewSize: CGSize,
                      panOffset: SIMD2<Float> = .zero,
                      zoomScale: Float = 1.0) -> [SIMD2<Float>] {
    var vertices: [SIMD2<Float>] = []

    guard centerPoints.count >= 2 else {
        if centerPoints.count == 1 {
            return createCircle(at: centerPoints[0],
                                radius: width / 2.0,
                                viewSize: viewSize,
                                panOffset: panOffset,
                                zoomScale: zoomScale)
        }
        return vertices
    }

    let halfWidth = Float(width / 2.0)

    // 1) START CAP
    let startCapVertices = createCircle(
        at: centerPoints[0],
        radius: width / 2.0,
        viewSize: viewSize,
        panOffset: panOffset,
        zoomScale: zoomScale
    )
    vertices.append(contentsOf: startCapVertices)

    // 2) SEGMENTS + JOINTS
    for i in 0..<(centerPoints.count - 1) {
        let current = centerPoints[i]
        let next = centerPoints[i + 1]

        let p1 = worldPixelToNDC(point: current, viewSize: viewSize, panOffset: panOffset, zoomScale: zoomScale)
        let p2 = worldPixelToNDC(point: next, viewSize: viewSize, panOffset: panOffset, zoomScale: zoomScale)

        let dir = p2 - p1
        let len = sqrt(dir.x * dir.x + dir.y * dir.y)
        guard len > 0 else { continue }
        let n = dir / len

        let perp = SIMD2<Float>(-n.y, n.x)

        let widthInNDC = (halfWidth / Float(viewSize.width)) * 2.0 * zoomScale

        let T1 = p1 + perp * widthInNDC
        let B1 = p1 - perp * widthInNDC
        let T2 = p2 + perp * widthInNDC
        let B2 = p2 - perp * widthInNDC

        vertices.append(T1); vertices.append(B1); vertices.append(T2)
        vertices.append(B1); vertices.append(B2); vertices.append(T2)

        if i < centerPoints.count - 2 {
            let jointVertices = createCircle(
                at: next,
                radius: width / 2.0,
                viewSize: viewSize,
                panOffset: panOffset,
                zoomScale: zoomScale,
                segments: 16
            )
            vertices.append(contentsOf: jointVertices)
        }
    }

    // 4) END CAP
    let endCapVertices = createCircle(
        at: centerPoints[centerPoints.count - 1],
        radius: width / 2.0,
        viewSize: viewSize,
        panOffset: panOffset,
        zoomScale: zoomScale
    )
    vertices.append(contentsOf: endCapVertices)

    return vertices
}

/// Triangle fan circle in NDC.
func createCircle(at point: CGPoint,
                  radius: CGFloat,
                  viewSize: CGSize,
                  panOffset: SIMD2<Float> = .zero,
                  zoomScale: Float = 1.0,
                  segments: Int = 30) -> [SIMD2<Float>] {
    var vertices: [SIMD2<Float>] = []

    let center = worldPixelToNDC(point: point, viewSize: viewSize, panOffset: panOffset, zoomScale: zoomScale)
    let radiusInNDC = (Float(radius) / Float(viewSize.width)) * 2.0 * zoomScale

    for i in 0..<segments {
        let a1 = Float(i) * (2.0 * .pi / Float(segments))
        let a2 = Float(i + 1) * (2.0 * .pi / Float(segments))

        let p1 = SIMD2<Float>(center.x + cos(a1) * radiusInNDC,
                              center.y + sin(a1) * radiusInNDC)
        let p2 = SIMD2<Float>(center.x + cos(a2) * radiusInNDC,
                              center.y + sin(a2) * radiusInNDC)

        vertices.append(center)
        vertices.append(p1)
        vertices.append(p2)
    }
    return vertices
}

// MARK: - Centerline Tessellation (Screen-Space Thickness)

/// Compute normalized distance values (0..1) for each point along the stroke
private func computeNormalizedU(points: [SIMD2<Float>]) -> [Float] {
    guard points.count > 1 else { return points.map { _ in 0 } }

    var lengths: [Float] = Array(repeating: 0, count: points.count)
    var totalLength: Float = 0
    var last = points[0]

    for i in 1..<points.count {
        let p = points[i]
        let dx = p.x - last.x
        let dy = p.y - last.y
        let d = sqrt(dx*dx + dy*dy)
        totalLength += d
        lengths[i] = totalLength
        last = p
    }

    // Avoid divide-by-zero
    if totalLength <= 0 {
        return points.map { _ in 0 }
    }

    // Normalize 0..1
    return lengths.map { $0 / totalLength }
}

/// Build a centerline-based vertex array where each centerline sample
/// produces two vertices: one for each side of the stroke band.
/// The GPU will extrude these to screen-space thickness in the vertex shader.
///
/// - parameter points: centerline points in stroke-local/world units.
/// - parameter color:  stroke color.
/// - returns: vertices suitable for a TRIANGLE_STRIP draw.
///
/// **Key Change from Full Tessellation:**
/// - Old: position = extruded vertex (includes width)
/// - New: position = centerline point, uv.y = side flag (-1 or +1)
/// - GPU uses uv.y to extrude in screen space, keeping thickness constant in pixels
///
/// **UV Coordinates:**
/// - uv.x: normalized position along stroke (0 at start, 1 at end) for round caps
/// - uv.y: side flag (-1 or +1) for extrusion direction
func tessellateCenterlineVertices(
    points: [SIMD2<Float>],
    color: SIMD4<Float>
) -> [StrokeVertex] {
    guard points.count >= 2 else { return [] }

    let uNorm = computeNormalizedU(points: points)

    var vertices: [StrokeVertex] = []
    vertices.reserveCapacity(points.count * 2)

    for i in 0..<points.count {
        let p = points[i]
        let u = uNorm[i] // 0 at start, 1 at end

        // side -1
        vertices.append(StrokeVertex(
            position: p,
            uv: SIMD2<Float>(u, -1),
            color: color
        ))

        // side +1
        vertices.append(StrokeVertex(
            position: p,
            uv: SIMD2<Float>(u, +1),
            color: color
        ))
    }

    return vertices
}

// MARK: - Local-Space Tessellation (Floating Origin Architecture)

/// Tessellate stroke in LOCAL space (relative to stroke origin).
/// This is view-agnostic: no NDC conversion, no zoom, no pan.
/// Just pure geometry: "offset these points by this width".
///
/// **Key Principle:** All inputs and outputs are in the same unit system
/// (world pixels), just centered around (0,0) instead of absolute world coords.
func tessellateStrokeLocal(centerPoints: [SIMD2<Float>],
                           width: Float) -> [SIMD2<Float>] {
    var vertices: [SIMD2<Float>] = []

    guard centerPoints.count >= 2 else {
        if centerPoints.count == 1 {
            return createCircleLocal(at: centerPoints[0], radius: width / 2.0)
        }
        return vertices
    }

    let halfWidth = width / 2.0

    // 1) START CAP
    let startCapVertices = createCircleLocal(at: centerPoints[0], radius: halfWidth)
    vertices.append(contentsOf: startCapVertices)

    // 2) SEGMENTS + JOINTS
    for i in 0..<(centerPoints.count - 1) {
        let p0 = centerPoints[i]
        let p1 = centerPoints[i + 1]

        // Direction vector
        let dir = p1 - p0
        let len = sqrt(dir.x * dir.x + dir.y * dir.y)
        guard len > 0 else { continue }

        let normalized = dir / len
        let perpendicular = SIMD2<Float>(-normalized.y, normalized.x)

        // Offset vertices by half-width
        let offset = perpendicular * halfWidth

        let top0 = p0 + offset
        let bot0 = p0 - offset
        let top1 = p1 + offset
        let bot1 = p1 - offset

        // Two triangles forming a quad
        vertices.append(top0)
        vertices.append(bot0)
        vertices.append(top1)

        vertices.append(bot0)
        vertices.append(bot1)
        vertices.append(top1)

        // Add joint circle at segment connections
        if i < centerPoints.count - 2 {
            let jointVertices = createCircleLocal(at: p1, radius: halfWidth, segments: 16)
            vertices.append(contentsOf: jointVertices)
        }
    }

    // 3) END CAP
    let endCapVertices = createCircleLocal(at: centerPoints[centerPoints.count - 1], radius: halfWidth)
    vertices.append(contentsOf: endCapVertices)

    return vertices
}

/// Create circle cap in local space.
/// Returns vertices relative to (0,0), not in NDC or screen space.
func createCircleLocal(at center: SIMD2<Float>,
                      radius: Float,
                      segments: Int = 30) -> [SIMD2<Float>] {
    var vertices: [SIMD2<Float>] = []

    for i in 0..<segments {
        let a1 = Float(i) * (2.0 * .pi / Float(segments))
        let a2 = Float(i + 1) * (2.0 * .pi / Float(segments))

        let p1 = SIMD2<Float>(center.x + cos(a1) * radius,
                              center.y + sin(a1) * radius)
        let p2 = SIMD2<Float>(center.x + cos(a2) * radius,
                              center.y + sin(a2) * radius)

        vertices.append(center)
        vertices.append(p1)
        vertices.append(p2)
    }

    return vertices
}

// MARK: - World-Space Tessellation (Double Precision - DEPRECATED)

/// Tessellate stroke in WORLD space using Double precision throughout.
/// This avoids float32 precision loss even with tiny world widths at extreme zoom.
/// Returns vertices in NDC at identity transform (ready for GPU transform).
func tessellateStrokeWorldDouble(centerPoints: [SIMD2<Double>],
                                 width: Double,
                                 viewSize: CGSize) -> [SIMD2<Float>] {
    var vertices: [SIMD2<Float>] = []

    guard centerPoints.count >= 2 else {
        if centerPoints.count == 1 {
            return createCircleWorldDouble(at: centerPoints[0],
                                          radius: width / 2.0,
                                          viewSize: viewSize)
        }
        return vertices
    }

    let halfWidth = width / 2.0

    // 1) START CAP
    let startCapVertices = createCircleWorldDouble(at: centerPoints[0],
                                                   radius: halfWidth,
                                                   viewSize: viewSize)
    vertices.append(contentsOf: startCapVertices)

    // 2) SEGMENTS + JOINTS
    for i in 0..<(centerPoints.count - 1) {
        let p0 = centerPoints[i]
        let p1 = centerPoints[i + 1]

        // Direction vector (Double precision)
        let dir = p1 - p0
        let len = sqrt(dir.x * dir.x + dir.y * dir.y)
        guard len > 0 else { continue }

        let normalized = dir / len
        let perpendicular = SIMD2<Double>(-normalized.y, normalized.x)

        // Offset vertices (Double precision)
        let offset = perpendicular * halfWidth

        let top0 = p0 + offset
        let bot0 = p0 - offset
        let top1 = p1 + offset
        let bot1 = p1 - offset

        // Convert to NDC (Double → Float only at the end)
        let T0 = worldToNDC_Double(top0, viewSize: viewSize)
        let B0 = worldToNDC_Double(bot0, viewSize: viewSize)
        let T1 = worldToNDC_Double(top1, viewSize: viewSize)
        let B1 = worldToNDC_Double(bot1, viewSize: viewSize)

        // Two triangles forming a quad
        vertices.append(T0)
        vertices.append(B0)
        vertices.append(T1)

        vertices.append(B0)
        vertices.append(B1)
        vertices.append(T1)

        // Add joint circle at segment connections
        if i < centerPoints.count - 2 {
            let jointVertices = createCircleWorldDouble(at: p1,
                                                       radius: halfWidth,
                                                       viewSize: viewSize,
                                                       segments: 16)
            vertices.append(contentsOf: jointVertices)
        }
    }

    // 3) END CAP
    let endCapVertices = createCircleWorldDouble(at: centerPoints[centerPoints.count - 1],
                                                 radius: halfWidth,
                                                 viewSize: viewSize)
    vertices.append(contentsOf: endCapVertices)

    return vertices
}

/// Create circle cap in world space using Double precision.
/// Returns vertices in NDC at identity transform.
func createCircleWorldDouble(at center: SIMD2<Double>,
                            radius: Double,
                            viewSize: CGSize,
                            segments: Int = 30) -> [SIMD2<Float>] {
    var vertices: [SIMD2<Float>] = []

    let centerNDC = worldToNDC_Double(center, viewSize: viewSize)

    // Radius in world units → NDC units (using Double)
    let W = Double(viewSize.width)
    let radiusNDC = Float((radius / W) * 2.0)

    for i in 0..<segments {
        let a1 = Double(i) * (2.0 * .pi / Double(segments))
        let a2 = Double(i + 1) * (2.0 * .pi / Double(segments))

        let p1 = SIMD2<Float>(centerNDC.x + Float(cos(a1)) * radiusNDC,
                              centerNDC.y + Float(sin(a1)) * radiusNDC)
        let p2 = SIMD2<Float>(centerNDC.x + Float(cos(a2)) * radiusNDC,
                              centerNDC.y + Float(sin(a2)) * radiusNDC)

        vertices.append(centerNDC)
        vertices.append(p1)
        vertices.append(p2)
    }

    return vertices
}

/// Convert world pixels to NDC using Double precision (at identity transform).
func worldToNDC_Double(_ world: SIMD2<Double>, viewSize: CGSize) -> SIMD2<Float> {
    let W = Double(viewSize.width)
    let H = Double(viewSize.height)

    // World pixels → NDC (using Double throughout)
    let x = (world.x / W) * 2.0 - 1.0
    let y = -((world.y / H) * 2.0 - 1.0)

    // Only convert to Float at the very end
    return SIMD2<Float>(Float(x), Float(y))
}

// MARK: - Screen-Space Tessellation (High Precision)

/// Create a circle in screen pixel coordinates (not NDC).
/// All geometry math happens in screen pixels to avoid precision loss.
func createCircleInScreenSpace(center: SIMD2<Float>,
                               radiusPixels: Float,
                               segments: Int = 30) -> [SIMD2<Float>] {
    var vertices: [SIMD2<Float>] = []

    for i in 0..<segments {
        let a1 = Float(i) * (2.0 * .pi / Float(segments))
        let a2 = Float(i + 1) * (2.0 * .pi / Float(segments))

        let p1 = SIMD2<Float>(center.x + cos(a1) * radiusPixels,
                              center.y + sin(a1) * radiusPixels)
        let p2 = SIMD2<Float>(center.x + cos(a2) * radiusPixels,
                              center.y + sin(a2) * radiusPixels)

        // Triangle fan
        vertices.append(center)
        vertices.append(p1)
        vertices.append(p2)
    }

    return vertices
}

/// Tessellate stroke in screen pixel coordinates.
/// All geometry calculations happen at screen-pixel scale to avoid precision issues.
/// Returns vertices in screen space - caller must convert to world space using double precision.
func tessellateStrokeInScreenSpace(screenPoints: [SIMD2<Float>],
                                   widthPixels: Float) -> [SIMD2<Float>] {
    var vertices: [SIMD2<Float>] = []

    guard screenPoints.count >= 2 else {
        if screenPoints.count == 1 {
            // Single point -> just a circle
            return createCircleInScreenSpace(center: screenPoints[0],
                                            radiusPixels: widthPixels * 0.5)
        }
        return vertices
    }

    let halfWidth = widthPixels * 0.5

    // 1) START CAP
    let startCapVertices = createCircleInScreenSpace(center: screenPoints[0],
                                                     radiusPixels: halfWidth)
    vertices.append(contentsOf: startCapVertices)

    // 2) SEGMENTS + JOINTS
    for i in 0..<(screenPoints.count - 1) {
        let p0 = screenPoints[i]
        let p1 = screenPoints[i + 1]

        // Calculate direction vector
        let dir = p1 - p0
        let len = sqrt(dir.x * dir.x + dir.y * dir.y)
        guard len > 0 else { continue }

        let normalized = dir / len
        let perpendicular = SIMD2<Float>(-normalized.y, normalized.x)

        // Offset vertices by half-width perpendicular to direction
        let offset = perpendicular * halfWidth

        let top0 = p0 + offset
        let bot0 = p0 - offset
        let top1 = p1 + offset
        let bot1 = p1 - offset

        // Two triangles forming a quad
        vertices.append(top0)
        vertices.append(bot0)
        vertices.append(top1)

        vertices.append(bot0)
        vertices.append(bot1)
        vertices.append(top1)

        // Add joint circle at segment connections (except at the last point)
        if i < screenPoints.count - 2 {
            let jointVertices = createCircleInScreenSpace(center: p1,
                                                         radiusPixels: halfWidth,
                                                         segments: 16)
            vertices.append(contentsOf: jointVertices)
        }
    }

    // 3) END CAP
    let endCapVertices = createCircleInScreenSpace(center: screenPoints[screenPoints.count - 1],
                                                   radiusPixels: halfWidth)
    vertices.append(contentsOf: endCapVertices)

    return vertices
}

// MARK: - Coordinate Conversion Helpers

/// Screen pixels -> World pixels (inverse of worldToScreenPixels).
//func screenToWorldPixels(_ p: CGPoint,
//                         viewSize: CGSize,
//                         panOffset: SIMD2<Float>,
//                         zoomScale: Float,
//                         rotationAngle: Float) -> CGPoint {
//    let cx = Float(viewSize.width) * 0.5
//    let cy = Float(viewSize.height) * 0.5
//
//    let sx = Float(p.x), sy = Float(p.y)
//
//    let centeredX = sx - cx
//    let centeredY = sy - cy
//
//    let unpannedX = centeredX - panOffset.x
//    let unpannedY = centeredY - panOffset.y
//
//    let unzoomedX = unpannedX / zoomScale
//    let unzoomedY = unpannedY / zoomScale
//
//
////    x = x'×cos(-θ) + y'×sin(-θ) = x'×cos(θ) - y'×sin(θ)
////    y = -x'×sin(-θ) + y'×cos(θ) = x'×sin(θ) + y'×cos(θ)
//
//    let unrotatedX = unzoomedX * cos(rotationAngle) - unzoomedY * sin(rotationAngle)
//    let unrotatedY = unzoomedX * sin(rotationAngle) + unzoomedY * cos(rotationAngle)
//
//    let wx = unrotatedX + cx
//    let wy = unrotatedY + cy
//
//    return CGPoint(x: CGFloat(wx), y: CGFloat(wy))
//}

// Screen pixels -> World pixels (full inverse of the shader path in *NDC*)
func screenToWorldPixels(_ p: CGPoint,
                         viewSize: CGSize,
                         panOffset: SIMD2<Float>,
                         zoomScale: Float,
                         rotationAngle: Float) -> CGPoint {
    let W = Float(viewSize.width)
    let H = Float(viewSize.height)

    // screen -> NDC
    let ndcX = (Float(p.x) / W) * 2.0 - 1.0
    let ndcY = -((Float(p.y) / H) * 2.0 - 1.0)

    // unpan in NDC (NOTE: same conversion/signs as shader)
    let panNDCx = (panOffset.x / W) * 2.0
    let panNDCy = -(panOffset.y / H) * 2.0
    let upX = ndcX - panNDCx
    let upY = ndcY - panNDCy

    // unzoom
    let uzX = upX / zoomScale
    let uzY = upY / zoomScale

    // unrotate (R(-θ))
    let c = cos(rotationAngle), s = sin(rotationAngle)
    let posX =  uzX * c - uzY * s
    let posY =  uzX * s + uzY * c

    // NDC -> world pixels (same mapping you use for vertices)
    let wx = ((posX + 1.0) * 0.5) * W
    let wy = ((1.0 - posY) * 0.5) * H
    return CGPoint(x: CGFloat(wx), y: CGFloat(wy))
}

// World pixels -> Screen pixels
func worldToScreenPixels(_ w: CGPoint,
                         viewSize: CGSize,
                         panOffset: SIMD2<Float>,
                         zoomScale: Float,
                         rotationAngle: Float) -> CGPoint {
    let W = Float(viewSize.width)
    let H = Float(viewSize.height)

    // world pixels -> model NDC (identity)
    let x0 = (Float(w.x) / W) * 2.0 - 1.0
    let y0 = -((Float(w.y) / H) * 2.0 - 1.0)

    // rotate (R(θ)), same as shader
    let c = cos(rotationAngle), s = sin(rotationAngle)
    let rx =  x0 * c + y0 * s
    let ry = -x0 * s + y0 * c

    // zoom
    let zx = rx * zoomScale
    let zy = ry * zoomScale

    // pan in NDC (same conversion/signs as shader)
    let panNDCx = (panOffset.x / W) * 2.0
    let panNDCy = -(panOffset.y / H) * 2.0
    let ndcX = zx + panNDCx
    let ndcY = zy + panNDCy

    // NDC -> screen pixels
    let sx = ((ndcX + 1.0) * 0.5) * W
    let sy = ((1.0 - ndcY) * 0.5) * H
    return CGPoint(x: CGFloat(sx), y: CGFloat(sy))

}

@inline(__always)
func worldToModelNDC(_ w: CGPoint, viewSize: CGSize) -> SIMD2<Float> {
    let W = Float(viewSize.width), H = Float(viewSize.height)
    let x = (Float(w.x) / W) * 2.0 - 1.0
    let y = -((Float(w.y) / H) * 2.0 - 1.0)
    return SIMD2<Float>(x, y)
}

@inline(__always)
func screenToNDC(_ p: CGPoint, viewSize: CGSize) -> SIMD2<Float> {
    let W = Float(viewSize.width), H = Float(viewSize.height)
    let x = (Float(p.x) / W) * 2.0 - 1.0
    let y = -((Float(p.y) / H) * 2.0 - 1.0)
    return SIMD2<Float>(x, y)
}

/// Convert screen pixels to world pixels using double precision.
/// This is used for high-precision conversion of tessellated geometry.
/// All intermediate calculations use Double to maintain precision even at extreme zoom levels.
func screenToWorldPixels_Double(_ p: SIMD2<Float>,
                                viewSize: CGSize,
                                panOffset: SIMD2<Float>,
                                zoomScale: Float,
                                rotationAngle: Float) -> SIMD2<Float> {
    // Use double precision for all intermediate calculations
    let W = Double(viewSize.width)
    let H = Double(viewSize.height)

    // Screen pixels -> NDC (using double)
    let ndcX = (Double(p.x) / W) * 2.0 - 1.0
    let ndcY = -((Double(p.y) / H) * 2.0 - 1.0)

    // Unpan in NDC (inverse of shader pan operation)
    let panNDCx = (Double(panOffset.x) / W) * 2.0
    let panNDCy = -(Double(panOffset.y) / H) * 2.0
    let upX = ndcX - panNDCx
    let upY = ndcY - panNDCy

    // Unzoom
    let zoom = Double(zoomScale)
    let uzX = upX / zoom
    let uzY = upY / zoom

    // Unrotate (R(-θ))
    let angle = Double(rotationAngle)
    let c = cos(angle)
    let s = sin(angle)
    let posX =  uzX * c - uzY * s
    let posY =  uzX * s + uzY * c

    // NDC -> world pixels
    let wx = ((posX + 1.0) * 0.5) * W
    let wy = ((1.0 - posY) * 0.5) * H

    return SIMD2<Float>(Float(wx), Float(wy))
}

///  PURE DOUBLE PRECISION (Pixel Space Rotation)
/// Converts Screen Pixels -> World Pixels avoiding NDC distortion.
/// Rotates in PIXEL SPACE to match the shader's pixel-space rotation.
func screenToWorldPixels_PureDouble(_ p: CGPoint,
                                    viewSize: CGSize,
                                    panOffset: SIMD2<Double>,
                                    zoomScale: Double,
                                    rotationAngle: Float) -> SIMD2<Double> {

    let center = SIMD2<Double>(Double(viewSize.width) / 2.0, Double(viewSize.height) / 2.0)
    let screenPt = SIMD2<Double>(Double(p.x), Double(p.y))

    // 1. Screen offset from center
    let offsetX = screenPt.x - center.x - panOffset.x
    let offsetY = screenPt.y - center.y - panOffset.y

    // 2. Unrotate (Inverse of Shader's CW Matrix)
    // Shader uses: [c, -s; s, c]
    // Inverse:     [c,  s; -s, c]
    let angle = Double(rotationAngle)
    let c = cos(angle)
    let s = sin(angle)

    let unrotatedX = offsetX * c + offsetY * s
    let unrotatedY = -offsetX * s + offsetY * c

    // 3. Unzoom
    let worldX = unrotatedX / zoomScale
    let worldY = unrotatedY / zoomScale

    return SIMD2<Double>(worldX, worldY)
}

/// Solve the pixel panOffset that keeps `anchorWorld` under `desiredScreen`
/// for the current zoom/rotation (matches shader math exactly).
func solvePanOffsetForAnchor(anchorWorld: SIMD2<Float>,
                             desiredScreen: CGPoint,
                             viewSize: CGSize,
                             zoomScale: Float,
                             rotationAngle: Float) -> SIMD2<Float> {
    let W = Float(viewSize.width), H = Float(viewSize.height)

    // modelNDC(w)
    let m = worldToModelNDC(CGPoint(x: CGFloat(anchorWorld.x), y: CGFloat(anchorWorld.y)),
                            viewSize: viewSize)

    // R(θ)*m
    let c = cos(rotationAngle), s = sin(rotationAngle)
    let rx =  m.x * c + m.y * s
    let ry = -m.x * s + m.y * c

    // desired screen → sNDC
    let sNDC = screenToNDC(desiredScreen, viewSize: viewSize)

    // panNDC = sNDC - zoom * R*m
    let panNDCx = sNDC.x - zoomScale * rx
    let panNDCy = sNDC.y - zoomScale * ry

    // Convert NDC pan to pixel panOffset (inverse of your shader’s conversion)
    // panNDC.x = (panPx / W)*2 → panPx = panNDC.x * (W/2)
    // panNDC.y = -(panPy / H)*2 → panPy = -panNDC.y * (H/2)
    let panPx = panNDCx * (W * 0.5)
    let panPy = -panNDCy * (H * 0.5)
    return SIMD2<Float>(panPx, panPy)
}

///  HIGH-PRECISION VERSION: Solve for Pan Offset using pure Double precision
/// This prevents stepping/stuttering at extreme zoom levels (1,000,000x+)
/// where Float precision (7 digits) is insufficient for smooth camera motion.
/// Works in PIXEL SPACE to match the new rotation logic.
func solvePanOffsetForAnchor_Double(anchorWorld: SIMD2<Double>,
                                    desiredScreen: CGPoint,
                                    viewSize: CGSize,
                                    zoomScale: Double,
                                    rotationAngle: Float) -> SIMD2<Double> {
    let center = SIMD2<Double>(Double(viewSize.width) / 2.0, Double(viewSize.height) / 2.0)

    // Forward transform: screen = (world * rot * zoom) + pan + center
    // Solve for pan: pan = screen - center - (world * rot * zoom)

    // 1. Rotate world point using CW matrix [c, -s; s, c]
    let angle = Double(rotationAngle)
    let c = cos(angle)
    let s = sin(angle)
    let rotatedX = anchorWorld.x * c - anchorWorld.y * s
    let rotatedY = anchorWorld.x * s + anchorWorld.y * c

    // 2. Zoom
    let zoomedX = rotatedX * zoomScale
    let zoomedY = rotatedY * zoomScale

    // 3. Solve for pan
    let panX = Double(desiredScreen.x) - center.x - zoomedX
    let panY = Double(desiredScreen.y) - center.y - zoomedY

    return SIMD2<Double>(panX, panY)
}



// MARK: - GPU Transform Struct (Floating Origin Architecture)

/// Legacy global transform (for reference, not used in floating origin system)
struct GPUTransform {
    var panOffset: SIMD2<Float>
    var zoomScale: Float
    var screenWidth: Float
    var screenHeight: Float
    var rotationAngle: Float
}
