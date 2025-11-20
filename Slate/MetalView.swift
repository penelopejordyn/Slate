import SwiftUI
import Metal
import MetalKit

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

// MARK: - Tile-Local Tessellation (Phase 2)

/// Create triangles for a stroke in tile-local space [0, 1024].
/// This works with tile-local coordinates and widths, avoiding float precision issues.
///
/// - Parameters:
///   - centerPoints: Points in tile-local space [0, 1024]
///   - width: Width in tile-local units (NOT world pixels!)
///   - tileSize: Size of tile in tile-local units (default 1024)
/// - Returns: Vertices in tile-local space ready for GPU transform
func tessellateStrokeLocal(centerPoints: [CGPoint],
                           width: CGFloat,
                           tileSize: CGFloat = 1024.0) -> [SIMD2<Float>] {
    var vertices: [SIMD2<Float>] = []

    guard centerPoints.count >= 2 else {
        if centerPoints.count == 1 {
            return createCircleLocal(at: centerPoints[0],
                                    radius: width / 2.0,
                                    tileSize: tileSize)
        }
        return vertices
    }

    let halfWidth = Float(width / 2.0)

    // 1) START CAP
    let startCapVertices = createCircleLocal(
        at: centerPoints[0],
        radius: width / 2.0,
        tileSize: tileSize
    )
    vertices.append(contentsOf: startCapVertices)

    // 2) SEGMENTS + JOINTS
    for i in 0..<(centerPoints.count - 1) {
        let current = centerPoints[i]
        let next = centerPoints[i + 1]

        // Points are already in tile-local space - use directly!
        let p1 = SIMD2<Float>(Float(current.x), Float(current.y))
        let p2 = SIMD2<Float>(Float(next.x), Float(next.y))

        let dir = p2 - p1
        let len = sqrt(dir.x * dir.x + dir.y * dir.y)
        guard len > 0 else { continue }
        let n = dir / len

        let perp = SIMD2<Float>(-n.y, n.x)

        // Width is already in tile-local units - use directly!
        let T1 = p1 + perp * halfWidth
        let B1 = p1 - perp * halfWidth
        let T2 = p2 + perp * halfWidth
        let B2 = p2 - perp * halfWidth

        vertices.append(T1); vertices.append(B1); vertices.append(T2)
        vertices.append(B1); vertices.append(B2); vertices.append(T2)

        if i < centerPoints.count - 2 {
            let jointVertices = createCircleLocal(
                at: next,
                radius: width / 2.0,
                tileSize: tileSize,
                segments: 16
            )
            vertices.append(contentsOf: jointVertices)
        }
    }

    // 3) END CAP
    let endCapVertices = createCircleLocal(
        at: centerPoints[centerPoints.count - 1],
        radius: width / 2.0,
        tileSize: tileSize
    )
    vertices.append(contentsOf: endCapVertices)

    return vertices
}

/// Triangle fan circle in tile-local space.
/// Returns vertices in tile-local coordinates [0, 1024].
///
/// - Parameters:
///   - point: Center point in tile-local space
///   - radius: Radius in tile-local units
///   - tileSize: Size of tile (default 1024)
///   - segments: Number of segments in circle (default 30)
/// - Returns: Triangle vertices in tile-local space
func createCircleLocal(at point: CGPoint,
                       radius: CGFloat,
                       tileSize: CGFloat = 1024.0,
                       segments: Int = 30) -> [SIMD2<Float>] {
    var vertices: [SIMD2<Float>] = []

    // Point and radius are already in tile-local space
    let center = SIMD2<Float>(Float(point.x), Float(point.y))
    let r = Float(radius)

    for i in 0..<segments {
        let a1 = Float(i) * (2.0 * .pi / Float(segments))
        let a2 = Float(i + 1) * (2.0 * .pi / Float(segments))

        let p1 = SIMD2<Float>(center.x + cos(a1) * r,
                              center.y + sin(a1) * r)
        let p2 = SIMD2<Float>(center.x + cos(a2) * r,
                              center.y + sin(a2) * r)

        vertices.append(center)
        vertices.append(p1)
        vertices.append(p2)
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

/// 🟢 PURE DOUBLE PRECISION: Screen pixels -> World pixels
/// Used for calculating Camera Center and Stroke Origins without truncation.
/// This is the FINAL precision fix - all inputs and outputs are Double.
func screenToWorldPixels_PureDouble(_ p: CGPoint,
                                    viewSize: CGSize,
                                    panOffset: SIMD2<Double>,
                                    zoomScale: Double,
                                    rotationAngle: Float) -> SIMD2<Double> {
    let W = Double(viewSize.width)
    let H = Double(viewSize.height)

    // 1. Screen -> NDC (Double)
    let ndcX = (Double(p.x) / W) * 2.0 - 1.0
    let ndcY = -((Double(p.y) / H) * 2.0 - 1.0)

    // 2. Unpan in NDC (using Double panOffset)
    // Note: panOffset is in screen pixels, so we convert it to NDC scaling
    let panNDCx = (panOffset.x / W) * 2.0
    let panNDCy = -(panOffset.y / H) * 2.0

    let upX = ndcX - panNDCx
    let upY = ndcY - panNDCy

    // 3. Unzoom (Double)
    let uzX = upX / zoomScale
    let uzY = upY / zoomScale

    // 4. Unrotate (R(-θ))
    let angle = Double(rotationAngle)
    let c = cos(angle)
    let s = sin(angle)
    let posX =  uzX * c - uzY * s
    let posY =  uzX * s + uzY * c

    // 5. NDC -> World Pixels
    let wx = ((posX + 1.0) * 0.5) * W
    let wy = ((1.0 - posY) * 0.5) * H

    return SIMD2<Double>(wx, wy)
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

/// 🟢 HIGH-PRECISION VERSION: Solve for Pan Offset using pure Double precision
/// This prevents stepping/stuttering at extreme zoom levels (1,000,000x+)
/// where Float precision (7 digits) is insufficient for smooth camera motion.
func solvePanOffsetForAnchor_Double(anchorWorld: SIMD2<Double>,
                                    desiredScreen: CGPoint,
                                    viewSize: CGSize,
                                    zoomScale: Double,
                                    rotationAngle: Float) -> SIMD2<Double> {
    let W = Double(viewSize.width)
    let H = Double(viewSize.height)

    // 1. World -> Model NDC (using Double)
    let mx = (anchorWorld.x / W) * 2.0 - 1.0
    let my = -((anchorWorld.y / H) * 2.0 - 1.0)

    // 2. Rotate: R(θ) * m
    let ang = Double(rotationAngle)
    let c = cos(ang)
    let s = sin(ang)
    let rx =  mx * c + my * s
    let ry = -mx * s + my * c

    // 3. Desired Screen -> Screen NDC
    let sNDCx = (Double(desiredScreen.x) / W) * 2.0 - 1.0
    let sNDCy = -((Double(desiredScreen.y) / H) * 2.0 - 1.0)

    // 4. Solve Pan in NDC: panNDC = sNDC - zoom * R*m
    let panNDCx = sNDCx - zoomScale * rx
    let panNDCy = sNDCy - zoomScale * ry

    // 5. NDC -> Pixel Pan (inverse of shader conversion)
    let panPx = panNDCx * (W * 0.5)
    let panPy = -panNDCy * (H * 0.5)

    return SIMD2<Double>(panPx, panPy)
}



// MARK: - GPU Transform Struct (Floating Origin Architecture)

/// Per-stroke transform using relative coordinates.
/// The GPU never sees absolute world coordinates - only small relative offsets.
struct StrokeTransform {
    var relativeOffset: SIMD2<Float>  // Stroke origin - Camera center (in world units)
    var zoomScale: Float              // Current zoom level
    var screenWidth: Float            // Screen dimensions
    var screenHeight: Float
    var rotationAngle: Float          // Camera rotation
}

/// Legacy global transform (for reference, not used in floating origin system)
struct GPUTransform {
    var panOffset: SIMD2<Float>
    var zoomScale: Float
    var screenWidth: Float
    var screenHeight: Float
    var rotationAngle: Float
}

// MARK: - MetalView

struct MetalView: UIViewRepresentable {
    func makeUIView(context: Context) -> MTKView {
        let mtkView = TouchableMTKView()
        mtkView.device = MTLCreateSystemDefaultDevice()
        mtkView.clearColor = MTLClearColor(red: 0.1, green: 1.0, blue: 1.0, alpha: 1.0)
        mtkView.delegate = context.coordinator
        mtkView.isUserInteractionEnabled = true

        mtkView.coordinator = context.coordinator
        context.coordinator.metalView = mtkView

        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator() }

    // MARK: - TouchableMTKView
    class TouchableMTKView: MTKView {
        weak var coordinator: Coordinator?

        var panGesture: UIPanGestureRecognizer!
        var pinchGesture: UIPinchGestureRecognizer!
        var rotationGesture: UIRotationGestureRecognizer!
        var longPressGesture: UILongPressGestureRecognizer!

        // 🟢 UPGRADED: Anchors now use Double for infinite precision at extreme zoom
        var pinchAnchorScreen: CGPoint = .zero
        var pinchAnchorWorld: SIMD2<Double> = .zero
        var panOffsetAtPinchStart: SIMD2<Double> = .zero

        //rotation anchor
        var rotationAnchorScreen: CGPoint = .zero
        var rotationAnchorWorld: SIMD2<Double> = .zero
        var panOffsetAtRotationStart: SIMD2<Double> = .zero

        enum AnchorOwner { case none, pinch, rotation }

        var activeOwner: AnchorOwner = .none
        var anchorWorld: SIMD2<Double> = .zero
        var anchorScreen: CGPoint = .zero
        
        var lastPinchTouchCount: Int = 0
        var lastRotationTouchCount: Int = 0


        
        
        func lockAnchor(owner: AnchorOwner, at screenPt: CGPoint, coord: Coordinator) {
            activeOwner = owner
            anchorScreen = screenPt
            // 🟢 Cast to Float temporarily for legacy screenToWorldPixels function
            let w = screenToWorldPixels(screenPt,
                                        viewSize: bounds.size,
                                        panOffset: SIMD2<Float>(Float(coord.panOffset.x), Float(coord.panOffset.y)),
                                        zoomScale: Float(coord.zoomScale),
                                        rotationAngle: coord.rotationAngle)
            // Store as Double for infinite precision
            anchorWorld = SIMD2<Double>(Double(w.x), Double(w.y))
        }

        // Re-lock anchor to a new screen point *without changing the transform*
        func relockAnchorAtCurrentCentroid(owner: AnchorOwner, screenPt: CGPoint, coord: Coordinator) {
            activeOwner = owner
            anchorScreen = screenPt
            // IMPORTANT: recompute world under the *new* centroid using current pan/zoom/rotation.
            // 🟢 Cast to Float temporarily for legacy screenToWorldPixels function
            let w = screenToWorldPixels(screenPt,
                                        viewSize: bounds.size,
                                        panOffset: SIMD2<Float>(Float(coord.panOffset.x), Float(coord.panOffset.y)),
                                        zoomScale: Float(coord.zoomScale),
                                        rotationAngle: coord.rotationAngle)
            // Store as Double for infinite precision
            anchorWorld = SIMD2<Double>(Double(w.x), Double(w.y))
        }

        func handoffAnchor(to newOwner: AnchorOwner, screenPt: CGPoint, coord: Coordinator) {
            relockAnchorAtCurrentCentroid(owner: newOwner, screenPt: screenPt, coord: coord)
        }

        func clearAnchorIfUnused() { activeOwner = .none }


        

        override init(frame: CGRect, device: MTLDevice?) {
            super.init(frame: frame, device: device)
            setupGestures()
        }
        required init(coder: NSCoder) {
            super.init(coder: coder)
            setupGestures()
        }

        func setupGestures() {
            panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
            panGesture.minimumNumberOfTouches = 2
            panGesture.maximumNumberOfTouches = 2
            addGestureRecognizer(panGesture)

            pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
            addGestureRecognizer(pinchGesture)

            rotationGesture = UIRotationGestureRecognizer(target: self, action: #selector(handleRotation(_:)))
            addGestureRecognizer(rotationGesture)

            longPressGesture = UILongPressGestureRecognizer(target: self, action: #selector(handleLongPress(_:)))
            longPressGesture.minimumPressDuration = 0.5
            longPressGesture.numberOfTouchesRequired = 2
            addGestureRecognizer(longPressGesture)

            panGesture.delegate = self
            pinchGesture.delegate = self
            rotationGesture.delegate = self
            longPressGesture.delegate = self
        }

        @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            guard let coord = coordinator else { return }
            let loc = gesture.location(in: self)
            let tc  = gesture.numberOfTouches

            switch gesture.state {
            case .began:
                lastPinchTouchCount = tc
                if activeOwner == .none { lockAnchor(owner: .pinch, at: loc, coord: coord) }

            case .changed:
                // If touch count changed (finger lifted/added), re-lock to new centroid w/o moving content.
                if activeOwner == .pinch, tc != lastPinchTouchCount {
                    relockAnchorAtCurrentCentroid(owner: .pinch, screenPt: loc, coord: coord)
                    lastPinchTouchCount = tc
                    // Do not solve pan on this exact frame; early-return to avoid visible snap.
                    gesture.scale = 1.0
                    return
                }

                // 🟢 Normal incremental zoom - multiply using Double precision
                coord.zoomScale = coord.zoomScale * Double(gesture.scale)
                gesture.scale = 1.0

                // 🟢 Keep the shared anchor pinned - use Double-precision solver
                let target = (activeOwner == .pinch) ? loc : anchorScreen
                coord.panOffset = solvePanOffsetForAnchor_Double(anchorWorld: anchorWorld,
                                                                 desiredScreen: target,
                                                                 viewSize: bounds.size,
                                                                 zoomScale: coord.zoomScale,
                                                                 rotationAngle: coord.rotationAngle)
                if activeOwner == .pinch { anchorScreen = target }

            case .ended, .cancelled, .failed:
                if activeOwner == .pinch {
                    // If rotation is active, hand off smoothly to its centroid
                    if rotationGesture.state == .changed || rotationGesture.state == .began {
                        let rloc = rotationGesture.location(in: self)
                        handoffAnchor(to: .rotation, screenPt: rloc, coord: coord)
                    } else {
                        clearAnchorIfUnused()
                    }
                }
                lastPinchTouchCount = 0

            default: break
            }
        }



        @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
            let t = gesture.translation(in: self)
            // 🟢 Add pan translation as Double for infinite precision
            coordinator?.panOffset.x += Double(t.x)
            coordinator?.panOffset.y += Double(t.y)
            gesture.setTranslation(.zero, in: self)

            if activeOwner != .none {
                anchorScreen.x += t.x
                anchorScreen.y += t.y
            }
        }

        
        @objc func handleRotation(_ gesture: UIRotationGestureRecognizer) {
            guard let coord = coordinator else { return }
            let loc = gesture.location(in: self)
            let tc  = gesture.numberOfTouches

            switch gesture.state {
            case .began:
                lastRotationTouchCount = tc
                if activeOwner == .none { lockAnchor(owner: .rotation, at: loc, coord: coord) }

            case .changed:
                // Re-lock when finger count changes to prevent jump.
                if activeOwner == .rotation, tc != lastRotationTouchCount {
                    relockAnchorAtCurrentCentroid(owner: .rotation, screenPt: loc, coord: coord)
                    lastRotationTouchCount = tc
                    gesture.rotation = 0.0
                    return
                }

                // Apply incremental rotation
                coord.rotationAngle += Float(gesture.rotation)
                gesture.rotation = 0.0

                // 🟢 Keep shared anchor pinned - use Double-precision solver
                let target = (activeOwner == .rotation) ? loc : anchorScreen
                coord.panOffset = solvePanOffsetForAnchor_Double(anchorWorld: anchorWorld,
                                                                 desiredScreen: target,
                                                                 viewSize: bounds.size,
                                                                 zoomScale: coord.zoomScale,
                                                                 rotationAngle: coord.rotationAngle)
                if activeOwner == .rotation { anchorScreen = target }

            case .ended, .cancelled, .failed:
                if activeOwner == .rotation {
                    if pinchGesture.state == .changed || pinchGesture.state == .began {
                        let ploc = pinchGesture.location(in: self)
                        handoffAnchor(to: .pinch, screenPt: ploc, coord: coord)
                    } else {
                        clearAnchorIfUnused()
                    }
                }
                lastRotationTouchCount = 0

            default: break
            }
        }

        @objc func handleLongPress(_ gesture: UILongPressGestureRecognizer) {
            guard let coord = coordinator else { return }

            if gesture.state == .began {
                coord.tileManager.debugMode.toggle()
                let status = coord.tileManager.debugMode ? "ON" : "OFF"
                print("\n🔧 TILE DEBUG MODE: \(status)")
                print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

                if coord.tileManager.debugMode {
                    let loc = gesture.location(in: self)
                    let worldPoint = screenToWorldPixels(loc,
                                                         viewSize: bounds.size,
                                                         panOffset: SIMD2<Float>(Float(coord.panOffset.x), Float(coord.panOffset.y)),
                                                         zoomScale: Float(coord.zoomScale),
                                                         rotationAngle: coord.rotationAngle)
                    let tileKey = coord.tileManager.getTileKey(worldPoint: worldPoint)
                    let debugOutput = coord.tileManager.debugInfo(worldPoint: worldPoint,
                                                                  screenPoint: loc,
                                                                  tileKey: tileKey)
                    print(debugOutput)

                    // PHASE 2 TEST: Tile-local tessellation
                    coord.testTileLocalTessellation()

                    // PHASE 3 DIAGNOSIS: Check for gaps issue
                    coord.diagnoseGapsIssue()
                }
            }
        }




        override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
            guard event?.allTouches?.count == 1, let touch = touches.first else { return }
            let location = touch.location(in: self)
            coordinator?.handleTouchBegan(at: location)
        }
        override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
            guard event?.allTouches?.count == 1, let touch = touches.first else { return }
            let location = touch.location(in: self)
            coordinator?.handleTouchMoved(at: location)
        }
        override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
            guard event?.allTouches?.count == 1, let touch = touches.first else { return }
            let location = touch.location(in: self)
            coordinator?.handleTouchEnded(at: location)
        }
        override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
            guard event?.allTouches?.count == 1 else { return }
            coordinator?.handleTouchCancelled()
        }
    }
}

// MARK: - Coordinator

class Coordinator: NSObject, MTKViewDelegate {
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var pipelineState: MTLRenderPipelineState!
    var vertexBuffer: MTLBuffer!

    var currentTouchPoints: [CGPoint] = []  // Stored in SCREEN space during drawing
    var liveStrokeOrigin: SIMD2<Double>?    // Temporary origin for live stroke (Double precision)
    var allStrokes: [Stroke] = []

    weak var metalView: MTKView?

    // 🟢 UPGRADED: Store camera state as Double for infinite precision
    var panOffset: SIMD2<Double> = .zero {
        didSet {
            // Cast to Float only when updating tile manager
            tileManager.currentPanOffset = SIMD2<Float>(Float(panOffset.x), Float(panOffset.y))
        }
    }
    var zoomScale: Double = 1.0 {
        didSet {
            // Cast to Float only when updating tile manager
            tileManager.currentZoomScale = Float(zoomScale)
        }
    }
    var rotationAngle: Float = 0.0

    // MARK: - Tiling System (Phase 1: Debug Only)
    let tileManager = TileManager()

    override init() {
        super.init()
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!
        makePipeLine()
        makeVertexBuffer()

        // Initialize tile manager state (cast Double to Float)
        tileManager.currentZoomScale = Float(zoomScale)
        tileManager.currentPanOffset = SIMD2<Float>(Float(panOffset.x), Float(panOffset.y))
    }

    /// Update tile manager with current view size (call when view size changes)
    func updateTileManagerViewSize(_ size: CGSize) {
        tileManager.viewSize = size
    }

    func draw(in view: MTKView) {
        let startTime = Date()

        // Update tile manager view size (only changes when view resizes)
        if tileManager.viewSize != view.bounds.size {
            tileManager.viewSize = view.bounds.size
        }

        // PHASE 2: Calculate Camera Center in World Space (Double precision)
        // This is the "View Center" - where the center of the screen is in the infinite world.
        let cameraCenterWorld = calculateCameraCenterWorld(viewSize: view.bounds.size)

        var allVertices: [SIMD2<Float>] = []
        var allTransforms: [StrokeTransform] = []

        // COMMITTED STROKES: Calculate relative offset for each stroke
        for stroke in allStrokes {
            guard !stroke.localVertices.isEmpty else { continue }

            // THE GREAT SUBTRACTION™ (in Double precision on CPU!)
            // This is the critical operation that prevents precision loss.
            let relativeOffsetDouble = stroke.origin - cameraCenterWorld
            let relativeOffset = SIMD2<Float>(Float(relativeOffsetDouble.x),
                                             Float(relativeOffsetDouble.y))

            // Create per-stroke transform
            // 🟢 Cast Double to Float only here, at the GPU boundary
            let transform = StrokeTransform(
                relativeOffset: relativeOffset,
                zoomScale: Float(zoomScale),
                screenWidth: Float(view.bounds.width),
                screenHeight: Float(view.bounds.height),
                rotationAngle: rotationAngle
            )

            allVertices.append(contentsOf: stroke.localVertices)
            allTransforms.append(transform)
        }

        // PHASE 5: LIVE STROKE with Screen-Space Deltas
        if currentTouchPoints.count >= 2, let tempOrigin = liveStrokeOrigin {
            // 🟢 NEW: Calculate geometry using screen-space deltas (infinite precision)
            // Formula: (ScreenPoint - FirstScreenPoint) / Zoom
            let firstScreenPt = currentTouchPoints[0]
            let zoom = Double(zoomScale)
            let cosAngle = Double(cos(-rotationAngle))
            let sinAngle = Double(sin(-rotationAngle))

            let localPoints = currentTouchPoints.map { pt in
                // Screen-space delta (high precision)
                let dx = Double(pt.x) - Double(firstScreenPt.x)
                let dy = Double(pt.y) - Double(firstScreenPt.y)

                // Un-rotate to match world space orientation
                let unrotatedX = dx * cosAngle - dy * sinAngle
                let unrotatedY = dx * sinAngle + dy * cosAngle

                // Convert to world units by dividing by zoom
                let worldDx = unrotatedX / zoom
                let worldDy = unrotatedY / zoom

                return SIMD2<Float>(Float(worldDx), Float(worldDy))
            }

            // Tessellate in LOCAL space
            let localVertices = tessellateStrokeLocal(
                centerPoints: localPoints,
                width: 10.0  // Constant 10px width in world units
            )

            // Calculate relative offset for live stroke
            let relativeOffsetDouble = tempOrigin - cameraCenterWorld
            let relativeOffset = SIMD2<Float>(Float(relativeOffsetDouble.x),
                                             Float(relativeOffsetDouble.y))

            // Create transform for live stroke
            // 🟢 Cast Double to Float only here, at the GPU boundary
            let liveTransform = StrokeTransform(
                relativeOffset: relativeOffset,
                zoomScale: Float(zoomScale),
                screenWidth: Float(view.bounds.width),
                screenHeight: Float(view.bounds.height),
                rotationAngle: rotationAngle
            )

            allVertices.append(contentsOf: localVertices)
            allTransforms.append(liveTransform)
        }

        let tessellationTime = Date().timeIntervalSince(startTime)
        if tessellationTime > 0.016 {
            print("⚠️ Tessellation taking \(String(format: "%.1f", tessellationTime * 1000))ms - \(allStrokes.count) strokes")
        }

        // PHASE 4: Render with per-stroke transforms
        guard !allVertices.isEmpty, !allTransforms.isEmpty else { return }

        let commandBuffer = commandQueue.makeCommandBuffer()!
        guard let rpd = view.currentRenderPassDescriptor else { return }
        let enc = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)!
        enc.setRenderPipelineState(pipelineState)
        enc.setCullMode(.none)

        // Render each stroke separately with its own transform
        var vertexOffset = 0
        for (index, stroke) in allStrokes.enumerated() {
            guard index < allTransforms.count else { break }
            let vertexCount = stroke.localVertices.count

            // Create vertex buffer for this stroke
            let vertexBuffer = device.makeBuffer(
                bytes: stroke.localVertices,
                length: vertexCount * MemoryLayout<SIMD2<Float>>.stride,
                options: .storageModeShared
            )

            // Create transform buffer for this stroke
            var transform = allTransforms[index]
            let transformBuffer = device.makeBuffer(
                bytes: &transform,
                length: MemoryLayout<StrokeTransform>.stride,
                options: .storageModeShared
            )

            enc.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            enc.setVertexBuffer(transformBuffer, offset: 0, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: vertexCount)

            vertexOffset += vertexCount
        }

        // Render live stroke if present
        if currentTouchPoints.count >= 2,
           liveStrokeOrigin != nil,
           allTransforms.count > allStrokes.count {
            // Extract live stroke vertices (last appended)
            let liveVertexCount = allVertices.count - vertexOffset
            let liveVertices = Array(allVertices.suffix(liveVertexCount))

            guard liveVertexCount > 0 else {
                enc.endEncoding()
                commandBuffer.present(view.currentDrawable!)
                commandBuffer.commit()
                return
            }

            let vertexBuffer = device.makeBuffer(
                bytes: liveVertices,
                length: liveVertexCount * MemoryLayout<SIMD2<Float>>.stride,
                options: .storageModeShared
            )

            var transform = allTransforms[allTransforms.count - 1]
            let transformBuffer = device.makeBuffer(
                bytes: &transform,
                length: MemoryLayout<StrokeTransform>.stride,
                options: .storageModeShared
            )

            enc.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            enc.setVertexBuffer(transformBuffer, offset: 0, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: liveVertexCount)
        }

        enc.endEncoding()
        commandBuffer.present(view.currentDrawable!)
        commandBuffer.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func makePipeLine() {
        let library = device.makeDefaultLibrary()!
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction   = library.makeFunction(name: "vertex_main")
        desc.fragmentFunction = library.makeFunction(name: "fragment_main")
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipelineState = try? device.makeRenderPipelineState(descriptor: desc)
    }

    func makeVertexBuffer() {
        var positions: [SIMD2<Float>] = [
            SIMD2<Float>(-0.8,  0.5),
            SIMD2<Float>(-0.3, -0.5),
            SIMD2<Float>(-0.8, -0.5),
            SIMD2<Float>(-0.3, -0.5),
            SIMD2<Float>(-0.3,  0.5),
            SIMD2<Float>(-0.8,  0.5),
        ]
        vertexBuffer = device.makeBuffer(bytes: &positions,
                                         length: positions.count * MemoryLayout<SIMD2<Float>>.stride,
                                         options: [])
    }

    func updateVertexBuffer(with vertices: [SIMD2<Float>]) {
        guard !vertices.isEmpty else { return }
        let bufferSize = vertices.count * MemoryLayout<SIMD2<Float>>.stride
        vertexBuffer = device.makeBuffer(bytes: vertices, length: bufferSize, options: .storageModeShared)
    }

    // MARK: - Camera Center Calculation

    /// Calculate the camera center in world coordinates using Double precision.
    /// This is the inverse of the pan/zoom/rotate transform applied to strokes.
    private func calculateCameraCenterWorld(viewSize: CGSize) -> SIMD2<Double> {
        // The center of the screen in screen coordinates
        let screenCenter = CGPoint(x: viewSize.width / 2.0, y: viewSize.height / 2.0)

        // 🟢 USE PURE DOUBLE HELPER
        // Pass Double panOffset and zoomScale directly without casting to Float
        // This prevents precision loss at extreme zoom levels (1,000,000x+)
        return screenToWorldPixels_PureDouble(screenCenter,
                                              viewSize: viewSize,
                                              panOffset: panOffset,       // Now passing SIMD2<Double>
                                              zoomScale: zoomScale,       // Now passing Double
                                              rotationAngle: rotationAngle)
    }

    // MARK: - Touch Handling

    func handleTouchBegan(at point: CGPoint) {
        guard let view = metalView else { return }

        // PHASE 5: Establish temporary origin for live stroke
        // The first touch point becomes the origin (in world coordinates, Double precision)
        let worldPoint = screenToWorldPixels(point,
                                            viewSize: view.bounds.size,
                                            panOffset: SIMD2<Float>(Float(panOffset.x), Float(panOffset.y)),
                                            zoomScale: Float(zoomScale),
                                            rotationAngle: rotationAngle)
        liveStrokeOrigin = SIMD2<Double>(Double(worldPoint.x), Double(worldPoint.y))

        // Keep points in SCREEN space during drawing
        currentTouchPoints = [point]

        // Debug tiling system (only when debug mode enabled)
        if tileManager.debugMode {
            let tileKey = tileManager.getTileKey(worldPoint: worldPoint)
            let debugOutput = tileManager.debugInfo(worldPoint: worldPoint,
                                                    screenPoint: point,
                                                    tileKey: tileKey)
            print("\n📍 TOUCH BEGAN - Temporary Origin: \(liveStrokeOrigin!)")
            print(debugOutput)
        }
    }

    func handleTouchMoved(at point: CGPoint) {
        // Keep points in SCREEN space during drawing (key for precision!)
        currentTouchPoints.append(point)

        // Debug tiling system (only when debug mode enabled, every 10th point)
        if tileManager.debugMode && currentTouchPoints.count % 10 == 0 {
            guard let view = metalView else { return }
            let worldPoint = screenToWorldPixels(point,
                                               viewSize: view.bounds.size,
                                               panOffset: SIMD2<Float>(Float(panOffset.x), Float(panOffset.y)),
                                               zoomScale: Float(zoomScale),
                                               rotationAngle: rotationAngle)
            let tileKey = tileManager.getTileKey(worldPoint: worldPoint)
            let debugOutput = tileManager.debugInfo(worldPoint: worldPoint,
                                                    screenPoint: point,
                                                    tileKey: tileKey)
            print("\n✏️ TOUCH MOVED (point \(currentTouchPoints.count))")
            print(debugOutput)
        }
    }

    func handleTouchEnded(at point: CGPoint) {
        guard let view = metalView else { return }

        // Keep final point in SCREEN space
        currentTouchPoints.append(point)

        // Debug tiling system (only when debug mode enabled)
        if tileManager.debugMode {
            let worldPoint = screenToWorldPixels(point,
                                               viewSize: view.bounds.size,
                                               panOffset: SIMD2<Float>(Float(panOffset.x), Float(panOffset.y)),
                                               zoomScale: Float(zoomScale),
                                               rotationAngle: rotationAngle)
            let tileKey = tileManager.getTileKey(worldPoint: worldPoint)
            let debugOutput = tileManager.debugInfo(worldPoint: worldPoint,
                                                    screenPoint: point,
                                                    tileKey: tileKey)
            print("\n🏁 TOUCH ENDED")
            print(debugOutput)
        }

        guard currentTouchPoints.count >= 4 else {
            currentTouchPoints = []
            liveStrokeOrigin = nil
            return
        }

        // Smooth the screen-space points
        let smoothScreenPoints = catmullRomPoints(points: currentTouchPoints,
                                                  closed: false,
                                                  alpha: 0.5,
                                                  segmentsPerCurve: 20)

        // 🟢 NEW: Pass SCREEN points directly to avoid Double precision loss
        // The stroke will calculate geometry using screen-space deltas: (pt - firstPt) / zoom
        // This preserves perfect smoothness at any zoom level
        let stroke = Stroke(screenPoints: smoothScreenPoints,
                            zoomAtCreation: zoomScale,
                            panAtCreation: panOffset,
                            viewSize: view.bounds.size,
                            rotationAngle: rotationAngle,
                            color: SIMD4<Float>(1.0, 0.0, 0.0, 1.0))

        allStrokes.append(stroke)
        currentTouchPoints = []
        liveStrokeOrigin = nil  // Clear temporary origin
    }

    func handleTouchCancelled() {
        currentTouchPoints = []
        liveStrokeOrigin = nil  // Clear temporary origin
    }

    // MARK: - Phase 3: Tile-Local Current Stroke Rendering

    /// Render current stroke using tile-local tessellation to avoid precision issues.
    /// This is the critical function that enables gap-free drawing at any zoom level!
    func renderCurrentStrokeTileLocal(view: MTKView) -> [SIMD2<Float>] {
        guard currentTouchPoints.count >= 2 else { return [] }

        // 1. Calculate width in tile-local space
        let widthTile = tileManager.calculateTileLocalWidth(baseWidth: 10.0)

        // 2. Group touch points by tile AND track which tiles each point belongs to
        var pointsByTile: [TileKey: [CGPoint]] = [:]
        var pointToTile: [Int: TileKey] = [:]  // Index → TileKey mapping

        for (index, worldPoint) in currentTouchPoints.enumerated() {
            let tileKey = tileManager.getTileKey(worldPoint: worldPoint)
            let localPoint = tileManager.worldToTileLocal(worldPoint: worldPoint, tileKey: tileKey)
            pointsByTile[tileKey, default: []].append(localPoint)
            pointToTile[index] = tileKey
        }

        // DEBUG: Check for tile boundary crossings
        if tileManager.debugMode {
            var crossings = 0
            for i in 0..<(currentTouchPoints.count - 1) {
                if pointToTile[i] != pointToTile[i + 1] {
                    crossings += 1
                    print("⚠️ TILE BOUNDARY CROSSING at segment \(i)→\(i+1)")
                    print("   Point \(i): tile \(pointToTile[i]?.description ?? "?")")
                    print("   Point \(i+1): tile \(pointToTile[i+1]?.description ?? "?")")
                }
            }

            if pointsByTile.count > 1 {
                print("📦 Stroke spans \(pointsByTile.count) tiles, \(crossings) boundary crossings")
                print("   Points per tile: \(pointsByTile.mapValues { $0.count })")
            }
        }

        // 3. Tessellate each tile's segment in tile-local space
        var allLocalVertices: [(tileKey: TileKey, vertices: [SIMD2<Float>])] = []

        for (tileKey, localPoints) in pointsByTile {
            guard localPoints.count >= 2 else { continue }

            let vertices = tessellateStrokeLocal(
                centerPoints: localPoints,
                width: CGFloat(widthTile)
            )
            allLocalVertices.append((tileKey: tileKey, vertices: vertices))
        }

        // 4. Convert tile-local vertices to world space, then to NDC
        var finalVertices: [SIMD2<Float>] = []

        for (tileKey, vertices) in allLocalVertices {
            for vertex in vertices {
                // Convert tile-local vertex to world coordinates
                let localPoint = CGPoint(x: Double(vertex.x), y: Double(vertex.y))
                let worldPoint = tileManager.tileLocalToWorld(localPoint: localPoint, tileKey: tileKey)

                // Convert world to NDC (existing pipeline)
                let ndcVertex = worldPixelToNDC(
                    point: worldPoint,
                    viewSize: view.bounds.size,
                    panOffset: SIMD2<Float>(Float(panOffset.x), Float(panOffset.y)),
                    zoomScale: Float(zoomScale)
                )

                finalVertices.append(ndcVertex)
            }
        }

        // Debug output for Phase 3
        if tileManager.debugMode && currentTouchPoints.count >= 2 {
            print("\n🎨 PHASE 3: TILE-LOCAL STROKE RENDERING")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("Zoom: \(zoomScale)x (level \(tileManager.referenceLevel))")
            print("Tile Size (world): \(String(format: "%.9f", tileManager.tileSizeAtReferenceLevel)) px")
            print("Width (world): \(String(format: "%.9f", 10.0 / Double(zoomScale))) px")
            print("Width (tile-local): \(String(format: "%.6f", widthTile)) units")
            print("Touch Points: \(currentTouchPoints.count)")
            print("Tiles Spanned: \(pointsByTile.count)")
            print("Total Vertices: \(finalVertices.count)")

            // Detailed point analysis
            print("\nPoint Analysis:")
            for (index, worldPoint) in currentTouchPoints.prefix(3).enumerated() {
                let tileKey = pointToTile[index]!
                let localPoint = tileManager.worldToTileLocal(worldPoint: worldPoint, tileKey: tileKey)

                print("  Point \(index):")
                print("    World: (\(String(format: "%.3f", worldPoint.x)), \(String(format: "%.3f", worldPoint.y)))")
                print("    Tile: Level \(tileKey.level), Grid (\(tileKey.tx), \(tileKey.ty))")
                print("    Local [0,1024]: (\(String(format: "%.3f", localPoint.x)), \(String(format: "%.3f", localPoint.y)))")

                // Check if local coords are in valid range
                if localPoint.x < 0 || localPoint.x > 1024 || localPoint.y < 0 || localPoint.y > 1024 {
                    print("    ⚠️ OUT OF RANGE!")
                } else {
                    print("    ✅ Valid")
                }
            }

            // Tessellation check
            print("\nTessellation Results:")
            for (tileKey, localPoints) in pointsByTile.prefix(3) {
                print("  Tile Level \(tileKey.level), Grid (\(tileKey.tx), \(tileKey.ty)): \(localPoints.count) points")
                if localPoints.count < 2 {
                    print("    ⚠️ INSUFFICIENT POINTS (need 2+)")
                }
            }

            // Coordinate magnitude check
            if let firstTile = allLocalVertices.first, let firstVertex = firstTile.vertices.first {
                print("\nCoordinate Magnitude Check:")
                print("  Tile-local vertex: (\(String(format: "%.3f", firstVertex.x)), \(String(format: "%.3f", firstVertex.y)))")
                let magnitude = max(abs(firstVertex.x), abs(firstVertex.y))
                print("  Max magnitude: \(String(format: "%.1f", magnitude))")

                if magnitude < 2000 {
                    print("  ✅ SMALL NUMBERS - Float precision preserved!")
                } else {
                    print("  ⚠️ LARGE NUMBERS - Float precision at risk!")
                }
            }
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        }

        return finalVertices
    }

    // MARK: - Phase 3 Diagnostics

    /// Diagnose gaps issue - test simple 2-point stroke
    func diagnoseGapsIssue() {
        print("\n🔬 GAP DIAGNOSIS")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        guard let view = metalView else { return }

        // Simulate a simple 2-point stroke at current zoom
        let screenP1 = CGPoint(x: 100, y: 100)
        let screenP2 = CGPoint(x: 200, y: 100)  // Horizontal line

        let worldP1 = screenToWorldPixels(screenP1, viewSize: view.bounds.size,
                                         panOffset: SIMD2<Float>(Float(panOffset.x), Float(panOffset.y)),
                                         zoomScale: Float(zoomScale),
                                         rotationAngle: rotationAngle)
        let worldP2 = screenToWorldPixels(screenP2, viewSize: view.bounds.size,
                                         panOffset: SIMD2<Float>(Float(panOffset.x), Float(panOffset.y)),
                                         zoomScale: Float(zoomScale),
                                         rotationAngle: rotationAngle)

        print("Test Stroke: Horizontal line, 100px on screen")
        print("  Screen P1: \(screenP1)")
        print("  Screen P2: \(screenP2)")
        print("  World P1: (\(String(format: "%.3f", worldP1.x)), \(String(format: "%.3f", worldP1.y)))")
        print("  World P2: (\(String(format: "%.3f", worldP2.x)), \(String(format: "%.3f", worldP2.y)))")
        print("  World distance: \(String(format: "%.3f", hypot(worldP2.x - worldP1.x, worldP2.y - worldP1.y))) px")

        // Check tile assignment
        let tile1 = tileManager.getTileKey(worldPoint: worldP1)
        let tile2 = tileManager.getTileKey(worldPoint: worldP2)
        let sameTile = (tile1 == tile2)

        print("\nTile Assignment:")
        print("  P1 tile: \(tile1.description)")
        print("  P2 tile: \(tile2.description)")
        print("  Same tile: \(sameTile ? "✅ YES" : "⚠️ NO - CROSSES BOUNDARY!")")

        // Test tile-local conversion
        let local1 = tileManager.worldToTileLocal(worldPoint: worldP1, tileKey: tile1)
        let local2 = tileManager.worldToTileLocal(worldPoint: worldP2, tileKey: tile2)

        print("\nTile-Local Coordinates:")
        print("  P1 local: (\(String(format: "%.3f", local1.x)), \(String(format: "%.3f", local1.y)))")
        print("  P2 local: (\(String(format: "%.3f", local2.x)), \(String(format: "%.3f", local2.y)))")

        if sameTile {
            let localDist = hypot(local2.x - local1.x, local2.y - local1.y)
            print("  Local distance: \(String(format: "%.3f", localDist))")

            // Test tessellation
            let widthTile = tileManager.calculateTileLocalWidth(baseWidth: 10.0)
            print("\nTessellation Test:")
            print("  Width (tile-local): \(String(format: "%.6f", widthTile))")

            let vertices = tessellateStrokeLocal(
                centerPoints: [local1, local2],
                width: CGFloat(widthTile)
            )
            print("  Generated vertices: \(vertices.count)")

            if vertices.isEmpty {
                print("  ⚠️ NO VERTICES GENERATED!")
            }
        } else {
            print("  ⚠️ Points in different tiles - segment will be LOST!")
        }

        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    }

    // MARK: - Phase 2 Testing

    /// Test tile-local tessellation with known inputs
    func testTileLocalTessellation() {
        print("\n🧪 TESTING TILE-LOCAL TESSELLATION")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        // Test case: Simple stroke in tile-local space
        let localPoints = [
            CGPoint(x: 100, y: 100),
            CGPoint(x: 200, y: 200),
            CGPoint(x: 300, y: 200)
        ]

        // Calculate tile-local width at current zoom
        let tileLocalWidth = tileManager.calculateTileLocalWidth(baseWidth: 10.0)

        print("Width Calculation:")
        print("  Zoom Scale: \(tileManager.currentZoomScale)")
        print("  Reference Level: \(tileManager.referenceLevel)")
        print("  Tile Size (world): \(String(format: "%.6f", tileManager.tileSizeAtReferenceLevel)) px")
        print("  World Width: \(String(format: "%.6f", 10.0 / Double(tileManager.currentZoomScale))) px")
        print("  Tile-Local Width: \(String(format: "%.6f", tileLocalWidth)) units")

        // Tessellate
        let vertices = tessellateStrokeLocal(
            centerPoints: localPoints,
            width: CGFloat(tileLocalWidth)
        )

        print("\nTessellation Results:")
        print("  Input Points: \(localPoints.count)")
        print("  Output Vertices: \(vertices.count)")

        if !vertices.isEmpty {
            let minX = vertices.map { $0.x }.min() ?? 0
            let maxX = vertices.map { $0.x }.max() ?? 0
            let minY = vertices.map { $0.y }.min() ?? 0
            let maxY = vertices.map { $0.y }.max() ?? 0

            print("  Vertex Range:")
            print("    X: [\(String(format: "%.2f", minX)), \(String(format: "%.2f", maxX))]")
            print("    Y: [\(String(format: "%.2f", minY)), \(String(format: "%.2f", maxY))]")

            // Check width offset from centerline
            let halfWidth = Float(tileLocalWidth / 2.0)
            print("  Expected Half-Width: \(String(format: "%.6f", halfWidth))")

            // Sample first quad (should be offset by halfWidth from p1)
            if vertices.count >= 6 {
                let p1 = SIMD2<Float>(Float(localPoints[0].x), Float(localPoints[0].y))
                let T1 = vertices[0]  // Top vertex of first segment
                let offset = distance(T1, p1)
                print("  Actual Offset from Center: \(String(format: "%.6f", offset))")

                let isCorrect = abs(offset - halfWidth) < 0.001
                print("  ✅ Width Correct: \(isCorrect)")
            }
        }

        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    }
}

// MARK: - Gesture Delegate

extension MetalView.TouchableMTKView: UIGestureRecognizerDelegate {
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                           shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        true
    }
}
