// Stroke.swift models pen and pencil strokes, including coordinate transforms and tessellation helpers.
import SwiftUI
import Metal

/// A compact, value-type node for cache-friendly traversal
struct FlatBVHNode {
    let bounds: CGRect
    let vertexStart: Int
    let vertexCount: Int
    let leftIndex: Int // Index in the flat array (-1 if leaf)
    let rightIndex: Int // Index in the flat array (-1 if leaf)
}

/// A stroke on the infinite canvas using Floating Origin architecture.
///
/// Local Realism:
/// Strokes are "locally aware" - they only know their position within the current Frame.
/// They are ignorant of the infinite universe and never see large numbers.
///
/// **Key Concept:** Instead of storing absolute world coordinates (which cause precision
/// issues at high zoom), we store:
/// - An `origin` (anchor point) within the current Frame (Double precision, but always small)
/// - All vertices as Float offsets from that origin (local coords)
///
/// This ensures the GPU only ever receives small Float values, eliminating precision gaps.
/// The stroke never needs to know if it exists at 10^100 zoom or 10^-50 zoom.
///
/// ** OPTIMIZATION: Buffer Caching**
/// Changed from struct to class to cache the GPU vertex buffer.
/// The buffer is created ONCE when the stroke is committed, eliminating
/// the CPU overhead of creating new buffers 60 times per second.
class Stroke: Identifiable {
    let id: UUID
    let origin: SIMD2<Double>           // Anchor point within the Frame (Double precision, always small)
    let localVertices: [StrokeVertex]   // Vertices with color baked in for batching
    let worldWidth: Double              // Width in world units
    let color: SIMD4<Float>

    //  OPTIMIZATION: Cached GPU Buffer
    // Created once in init, reused every frame
    var vertexBuffer: MTLBuffer?

    //  OPTIMIZATION: Bounding Box for Culling
    // Calculated once in init, used to skip off-screen strokes
    var localBounds: CGRect = .zero

    //  OPTIMIZATION: Flat Array BVH for Cache-Friendly Culling
    // Linear memory layout allows CPU pre-fetching, drastically reducing frame time
    // Root is at the end of the array, leaves are 32-vertex chunks
    var flatNodes: [FlatBVHNode] = []

    /// Initialize stroke from screen-space points using direct delta calculation.
    /// This avoids Double precision loss at extreme zoom levels by calculating
    /// the stroke geometry directly from screen-space deltas rather than converting
    /// all points to absolute world coordinates.
    ///
    /// - Parameters:
    ///   - screenPoints: Raw screen points (high precision)
    ///   - zoomAtCreation: Zoom level when stroke was drawn
    ///   - panAtCreation: Pan offset when stroke was drawn
    ///   - viewSize: View dimensions
    ///   - rotationAngle: Camera rotation angle
    ///   - color: Stroke color
    ///   - baseWidth: Base stroke width in world units (before zoom adjustment)
    ///   - device: MTLDevice for creating cached vertex buffer
    ///  UPGRADED: Now accepts Double for zoom and pan to maintain precision
    ///  OPTIMIZATION: Now caches vertex buffer and bounding box in init
    init(screenPoints: [CGPoint],
         zoomAtCreation: Double,
         panAtCreation: SIMD2<Double>,
         viewSize: CGSize,
         rotationAngle: Float,
         color: SIMD4<Float>,
         baseWidth: Double = 10.0,
         device: MTLDevice?) {
        self.id = UUID()
        self.color = color

        guard let firstScreenPt = screenPoints.first else {
            self.origin = .zero
            self.localVertices = []
            self.worldWidth = 0
            return
        }

        // 1. CALCULATE ORIGIN (ABSOLUTE) -  HIGH PRECISION FIX
        // We still need the absolute world position for the anchor, so we know WHERE the stroke is.
        // Only convert the FIRST point to world coordinates.
        // Use the Pure Double helper so the anchor is precise at 1,000,000x zoom
        self.origin = screenToWorldPixels_PureDouble(firstScreenPt,
                                                     viewSize: viewSize,
                                                     panOffset: panAtCreation,
                                                     zoomScale: zoomAtCreation,
                                                     rotationAngle: rotationAngle)

        // 2. CALCULATE GEOMETRY (RELATIVE) - THE FIX for Double precision
        //  Calculate shape directly from screen deltas: (ScreenPoint - FirstScreenPoint) / Zoom
        // This preserves perfect smoothness regardless of world coordinates.
        let zoom = zoomAtCreation
        let angle = Double(rotationAngle)
        let c = cos(angle)
        let s = sin(angle)

        let relativePoints: [SIMD2<Float>] = screenPoints.map { pt in
            let dx = Double(pt.x) - Double(firstScreenPt.x)
            let dy = Double(pt.y) - Double(firstScreenPt.y)

            //  FIX: Match the CPU Inverse Rotation (Screen -> World)
            // Inverse of Shader's CW matrix: [c, s; -s, c]
            let unrotatedX = dx * c + dy * s
            let unrotatedY = -dx * s + dy * c

            // Convert to world units
            let worldDx = unrotatedX / zoom
            let worldDy = unrotatedY / zoom

            return SIMD2<Float>(Float(worldDx), Float(worldDy))
        }

        // 2.5. Clamp center points to prevent excessive vertex counts
        var centerPoints = relativePoints

        let maxCenterPoints = 1000  // Maximum number of centerline points per stroke
        if centerPoints.count > maxCenterPoints {
            let step = max(1, centerPoints.count / maxCenterPoints)
            var downsampled: [SIMD2<Float>] = []
            downsampled.reserveCapacity(maxCenterPoints + 1)

            for i in stride(from: 0, to: centerPoints.count, by: step) {
                downsampled.append(centerPoints[i])
            }
            if let last = centerPoints.last, last != downsampled.last {
                downsampled.append(last)
            }

            centerPoints = downsampled
        }

        // 3. World width is the base width divided by zoom
        let worldWidth = baseWidth / zoom
        self.worldWidth = worldWidth

        // 4. Tessellate in LOCAL space using CENTERLINE approach
        // New: GPU will extrude to screen-space thickness, keeping strokes sharp at all zoom levels
        let vertices = tessellateCenterlineVertices(
            points: centerPoints,
            color: color
        )

        // 4.5. Safety net: prevent excessive vertex counts
        let maxVertices = 60_000
        if vertices.count > maxVertices {
            // Drop stroke if it would be too expensive to render
            self.localVertices = []
            self.vertexBuffer = nil
            self.localBounds = .null
            self.flatNodes = []
            return
        }

        self.localVertices = vertices

        // 5.  OPTIMIZATION: Create Cached Buffer
        // This is done ONCE here, not 60 times per second in drawStroke
        if let device = device, !localVertices.isEmpty {
            self.vertexBuffer = device.makeBuffer(
                bytes: localVertices,
                length: localVertices.count * MemoryLayout<StrokeVertex>.stride,
                options: .storageModeShared
            )
        }

        // 6.  OPTIMIZATION: Build Linear BVH for Cache-Friendly Culling
        // Flat array structure eliminates pointer chasing and fits in CPU cache
        if !localVertices.isEmpty {
            // Reserve capacity to avoid allocations during build
            // A binary tree has approx 2*N nodes where N is number of leaves
            let leafCount = (localVertices.count / 32) + 1
            flatNodes.reserveCapacity(leafCount * 2)

            let rootIndex = buildLinearBVH(start: 0, count: localVertices.count)
            if rootIndex >= 0 {
                self.localBounds = flatNodes[rootIndex].bounds
            }
        }
    }

    /// Recursive builder that populates the flat array
    /// Returns the index of the node created
    /// Uses 32-vertex leaves for balanced performance between culling precision and tree depth
    private func buildLinearBVH(start: Int, count: Int) -> Int {
        // LEAF CASE (32 vertices = 16 triangles)
        // Larger leaves reduce tree depth and traversal overhead
        if count <= 32 {
            let bounds = calculateBounds(start: start, count: count)
            let node = FlatBVHNode(bounds: bounds, vertexStart: start, vertexCount: count, leftIndex: -1, rightIndex: -1)
            flatNodes.append(node)
            return flatNodes.count - 1
        }

        // INTERNAL CASE: Split in half
        // Children must be created first so their indices are valid
        let mid = count / 2
        let leftIdx = buildLinearBVH(start: start, count: mid)
        let rightIdx = buildLinearBVH(start: start + mid, count: count - mid)

        // Parent bounds is the union of children
        let totalBounds = flatNodes[leftIdx].bounds.union(flatNodes[rightIdx].bounds)
        let node = FlatBVHNode(bounds: totalBounds, vertexStart: 0, vertexCount: 0, leftIndex: leftIdx, rightIndex: rightIdx)
        flatNodes.append(node)
        return flatNodes.count - 1
    }

    private func calculateBounds(start: Int, count: Int) -> CGRect {
        var minX = Float.greatestFiniteMagnitude
        var maxX = -Float.greatestFiniteMagnitude
        var minY = Float.greatestFiniteMagnitude
        var maxY = -Float.greatestFiniteMagnitude

        let end = min(start + count, localVertices.count)
        for i in start..<end {
            let pos = localVertices[i].position
            if pos.x < minX { minX = pos.x }
            if pos.x > maxX { maxX = pos.x }
            if pos.y < minY { minY = pos.y }
            if pos.y > maxY { maxY = pos.y }
        }

        // Expand bounds by stroke half-width (centerline vertices need expansion for GPU extrusion)
        // Since we're using centerline vertices, the positions are at the centerline
        // We need to expand by the maximum stroke radius for accurate culling
        let halfWidth = Float(worldWidth) * 0.5
        minX -= halfWidth
        maxX += halfWidth
        minY -= halfWidth
        maxY += halfWidth

        return CGRect(x: Double(minX), y: Double(minY), width: Double(maxX - minX), height: Double(maxY - minY))
    }
}
