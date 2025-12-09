// Stroke.swift models pen and pencil strokes, including coordinate transforms and tessellation helpers.
import SwiftUI
import Metal

/// A compact, value-type node for cache-friendly traversal
struct FlatBVHNode {
    let bounds: CGRect
    let segmentStart: Int
    let segmentCount: Int
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
    let centerline: [SIMD2<Float>]      // Raw center points in local space for BVH construction
    let localVertices: [StrokeVertex]   // Vertices with color baked in for batching
    let worldWidth: Double              // Width in world units
    let color: SIMD4<Float>
    var primitiveType: MTLPrimitiveType

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
    var bvhRootIndex: Int?

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
        self.primitiveType = .triangleStrip

        guard let firstScreenPt = screenPoints.first else {
            self.origin = .zero
            self.centerline = []
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

        self.centerline = relativePoints

        // 3. World width is the base width divided by zoom
        let worldWidth = baseWidth / zoom
        self.worldWidth = worldWidth

        // 4. Tessellate in LOCAL space (no view-specific transforms)
        self.localVertices = buildStrokeStripVertices(
            centerPoints: relativePoints,
            width: Float(worldWidth),
            color: color
        )

        // 4.5 Primitive type depends on tessellation output
        if relativePoints.count < 2 {
            self.primitiveType = .triangle
        }

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
        let segmentCount = max(centerline.count - 1, 0)
        if segmentCount > 0 {
            // Reserve capacity to avoid allocations during build
            let leafCount = (segmentCount / 8) + 1
            flatNodes.reserveCapacity(leafCount * 2)

            let rootIndex = buildLinearBVH(segmentStart: 0, segmentCount: segmentCount)
            if rootIndex >= 0 {
                self.localBounds = flatNodes[rootIndex].bounds
                self.bvhRootIndex = rootIndex
            }
        } else if let first = centerline.first {
            let radius = Float(worldWidth * 0.5)
            self.localBounds = CGRect(x: Double(first.x - radius),
                                      y: Double(first.y - radius),
                                      width: Double(radius * 2),
                                      height: Double(radius * 2))
        }
    }

    /// Recursive builder that populates the flat array
    /// Returns the index of the node created
    /// Uses small segment strips for balanced precision and tree depth
    private func buildLinearBVH(segmentStart: Int, segmentCount: Int) -> Int {
        // LEAF CASE (small segment batches)
        if segmentCount <= 8 {
            let bounds = calculateBounds(startSegment: segmentStart, segmentCount: segmentCount)
            let node = FlatBVHNode(bounds: bounds, segmentStart: segmentStart, segmentCount: segmentCount, leftIndex: -1, rightIndex: -1)
            flatNodes.append(node)
            return flatNodes.count - 1
        }

        // INTERNAL CASE: Split in half
        let mid = segmentCount / 2
        let leftIdx = buildLinearBVH(segmentStart: segmentStart, segmentCount: mid)
        let rightIdx = buildLinearBVH(segmentStart: segmentStart + mid, segmentCount: segmentCount - mid)

        // Parent bounds is the union of children
        let totalBounds = flatNodes[leftIdx].bounds.union(flatNodes[rightIdx].bounds)
        let node = FlatBVHNode(bounds: totalBounds, segmentStart: 0, segmentCount: 0, leftIndex: leftIdx, rightIndex: rightIdx)
        flatNodes.append(node)
        return flatNodes.count - 1
    }

    private func calculateBounds(startSegment: Int, segmentCount: Int) -> CGRect {
        var minX = Float.greatestFiniteMagnitude
        var maxX = -Float.greatestFiniteMagnitude
        var minY = Float.greatestFiniteMagnitude
        var maxY = -Float.greatestFiniteMagnitude

        let startPoint = startSegment
        let endPoint = min(startSegment + segmentCount, centerline.count - 1)

        for i in startPoint...(endPoint) {
            let pos = centerline[i]
            if pos.x < minX { minX = pos.x }
            if pos.x > maxX { maxX = pos.x }
            if pos.y < minY { minY = pos.y }
            if pos.y > maxY { maxY = pos.y }
        }

        let halfWidth = Float(worldWidth * 0.5)
        minX -= halfWidth
        maxX += halfWidth
        minY -= halfWidth
        maxY += halfWidth

        return CGRect(x: Double(minX), y: Double(minY), width: Double(maxX - minX), height: Double(maxY - minY))
    }

    /// Traverse the BVH and return contiguous vertex ranges that intersect the culling radius.
    /// Leaf ranges are aligned to the two-vertex-per-point strip format for contiguous draws.
    func visibleVertexRanges(relativeOffset: SIMD2<Double>,
                             zoomScale: Double,
                             cullRadius: Double,
                             onVisibleLeaf: ((Int) -> Void)? = nil) -> [(Int, Int)] {
        guard !localVertices.isEmpty else { return [] }

        // Early out using whole-stroke bounds
        let boundsCenter = SIMD2<Double>(localBounds.midX, localBounds.midY)
        let worldCenter = relativeOffset + boundsCenter
        let distWorld = sqrt(worldCenter.x * worldCenter.x + worldCenter.y * worldCenter.y)
        let distScreen = distWorld * zoomScale

        let strokeRadiusWorld = sqrt(pow(localBounds.width, 2) + pow(localBounds.height, 2)) * 0.5
        let strokeRadiusScreen = strokeRadiusWorld * zoomScale

        if (distScreen - strokeRadiusScreen) > cullRadius {
            return []
        }

        guard let rootIndex = bvhRootIndex else {
            // No BVH (likely a point stroke) - draw everything as a single leaf.
            if localVertices.count >= 3 {
                onVisibleLeaf?(localVertices.count)
                return [(0, localVertices.count)]
            }
            return []
        }

        var ranges: [(Int, Int)] = []
        var stack: [Int] = [rootIndex]

        while let nodeIndex = stack.popLast() {
            let node = flatNodes[nodeIndex]
            let nodeCenter = SIMD2<Double>(node.bounds.midX, node.bounds.midY)
            let nodeWorldCenter = relativeOffset + nodeCenter
            let nodeDistWorld = sqrt(nodeWorldCenter.x * nodeWorldCenter.x + nodeWorldCenter.y * nodeWorldCenter.y)
            let nodeDistScreen = nodeDistWorld * zoomScale
            let nodeRadiusWorld = sqrt(pow(node.bounds.width, 2) + pow(node.bounds.height, 2)) * 0.5
            let nodeRadiusScreen = nodeRadiusWorld * zoomScale

            if (nodeDistScreen - nodeRadiusScreen) > cullRadius {
                continue
            }

            // Leaf
            if node.leftIndex == -1 && node.rightIndex == -1 {
                let vertexStart = node.segmentStart * 2
                let vertexCount = (node.segmentCount + 1) * 2
                if vertexCount >= 3 {
                    onVisibleLeaf?(vertexCount)
                    ranges.append((vertexStart, vertexCount))
                }
            } else {
                if node.leftIndex != -1 { stack.append(node.leftIndex) }
                if node.rightIndex != -1 { stack.append(node.rightIndex) }
            }
        }

        return ranges
    }
}
