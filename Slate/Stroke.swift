// Stroke.swift models pen and pencil strokes using distance-to-segment SDF instancing.
import SwiftUI
import Metal

/// A stroke on the infinite canvas using Floating Origin architecture.
///
/// Local Realism:
/// Strokes are "locally aware" - they only know their position within the current Frame.
/// They are ignorant of the infinite universe and never see large numbers.
///
/// **Key Concept:** Instead of storing absolute world coordinates (which cause precision
/// issues at high zoom), we store:
/// - An `origin` (anchor point) within the current Frame (Double precision, but always small)
/// - All segment endpoints as Float offsets from that origin (local coords)
class Stroke: Identifiable {
    let id: UUID
    let origin: SIMD2<Double>           // Anchor point within the Frame (Double precision, always small)
    let worldWidth: Double              // Width in world units
    let color: SIMD4<Float>
    let zoomEffectiveAtCreation: Float  // Effective zoom when the stroke was committed
    let depthID: UInt32                // Global draw order for depth testing (larger = newer)
    var depthWriteEnabled: Bool         // Allows disabling depth writes for specific strokes (e.g. translucent markers)

    /// GPU segment instances for SDF rendering
    let segments: [StrokeSegmentInstance]
    var segmentBuffer: MTLBuffer?

    /// Bounding box in local space for culling
    var localBounds: CGRect = .zero
    /// Optional duplicate for future BVH/tiling
    var segmentBounds: CGRect = .zero

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
    ///   - zoomEffectiveAtCreation: Effective zoom at commit time for disappearance logic
    ///   - device: MTLDevice for creating cached vertex buffer
    ///  UPGRADED: Now accepts Double for zoom and pan to maintain precision
    ///  OPTIMIZATION: Now caches GPU segment buffer in init
    init(screenPoints: [CGPoint],
         zoomAtCreation: Double,
         panAtCreation: SIMD2<Double>,
         viewSize: CGSize,
         rotationAngle: Float,
         color: SIMD4<Float>,
         baseWidth: Double = 10.0,
         zoomEffectiveAtCreation: Float,
         device: MTLDevice?,
         depthID: UInt32,
         depthWriteEnabled: Bool = true) {
        self.id = UUID()
        self.color = color
        self.depthID = depthID
        let safeZoom = max(zoomAtCreation, 1e-6)
        self.zoomEffectiveAtCreation = max(zoomEffectiveAtCreation, 1e-6)
        self.depthWriteEnabled = depthWriteEnabled

        guard let firstScreenPt = screenPoints.first else {
            self.origin = .zero
            self.segments = []
            self.worldWidth = 0
            self.localBounds = .null
            self.segmentBounds = .null
            return
        }

        // 1. CALCULATE ORIGIN (ABSOLUTE) -  HIGH PRECISION FIX
        self.origin = screenToWorldPixels_PureDouble(firstScreenPt,
                                                     viewSize: viewSize,
                                                     panOffset: panAtCreation,
                                                     zoomScale: safeZoom,
                                                     rotationAngle: rotationAngle)

        // 2. CALCULATE GEOMETRY (RELATIVE) - THE FIX for Double precision
        //  Calculate shape directly from screen deltas: (ScreenPoint - FirstScreenPoint) / Zoom
        let zoom = safeZoom
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

        // 2.5. Clamp center points to prevent excessive counts
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

        // 4. Build segment instances
        let builtSegments = Stroke.buildSegments(from: centerPoints, color: color)
        self.segments = builtSegments

        // 5. Calculate bounds expanded by stroke radius for culling
        let bounds = Stroke.calculateBounds(for: centerPoints, radius: Float(worldWidth) * 0.5)
        self.localBounds = bounds
        self.segmentBounds = bounds

        // 6. Create cached segment buffer
        if let device = device, !builtSegments.isEmpty {
            self.segmentBuffer = device.makeBuffer(
                bytes: builtSegments,
                length: builtSegments.count * MemoryLayout<StrokeSegmentInstance>.stride,
                options: .storageModeShared
            )
        }
    }
}

// MARK: - Helpers
extension Stroke {
    static func buildSegments(from points: [SIMD2<Float>], color: SIMD4<Float>) -> [StrokeSegmentInstance] {
        guard !points.isEmpty else { return [] }

        if points.count == 1 {
            let p = points[0]
            return [StrokeSegmentInstance(p0: p, p1: p, color: color)]
        }

        var segments: [StrokeSegmentInstance] = []
        segments.reserveCapacity(points.count - 1)

        for i in 0..<(points.count - 1) {
            let p0 = points[i]
            let p1 = points[i + 1]
            segments.append(StrokeSegmentInstance(p0: p0, p1: p1, color: color))
        }

        return segments
    }

    static func calculateBounds(for points: [SIMD2<Float>], radius: Float) -> CGRect {
        guard let first = points.first else { return .null }

        var minX = first.x
        var maxX = first.x
        var minY = first.y
        var maxY = first.y

        for p in points.dropFirst() {
            minX = min(minX, p.x)
            maxX = max(maxX, p.x)
            minY = min(minY, p.y)
            maxY = max(maxY, p.y)
        }

        minX -= radius
        maxX += radius
        minY -= radius
        maxY += radius

        return CGRect(x: Double(minX), y: Double(minY), width: Double(maxX - minX), height: Double(maxY - minY))
    }
}
