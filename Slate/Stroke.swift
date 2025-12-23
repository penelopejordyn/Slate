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

    init(id: UUID = UUID(),
         origin: SIMD2<Double>,
         worldWidth: Double,
         color: SIMD4<Float>,
         zoomEffectiveAtCreation: Float,
         segments: [StrokeSegmentInstance],
         localBounds: CGRect,
         segmentBounds: CGRect? = nil,
         device: MTLDevice?,
         depthID: UInt32,
         depthWriteEnabled: Bool = true) {
        self.id = id
        self.origin = origin
        self.worldWidth = worldWidth
        self.color = color
        self.zoomEffectiveAtCreation = max(zoomEffectiveAtCreation, 1e-6)
        self.depthID = depthID
        self.depthWriteEnabled = depthWriteEnabled
        self.segments = segments
        self.localBounds = localBounds
        self.segmentBounds = segmentBounds ?? localBounds

        if let device = device, !segments.isEmpty {
            self.segmentBuffer = device.makeBuffer(
                bytes: segments,
                length: segments.count * MemoryLayout<StrokeSegmentInstance>.stride,
                options: .storageModeShared
            )
        }
    }

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
    convenience init(screenPoints: [CGPoint],
                     zoomAtCreation: Double,
                     panAtCreation: SIMD2<Double>,
                     viewSize: CGSize,
                     rotationAngle: Float,
                     color: SIMD4<Float>,
                     baseWidth: Double = 10.0,
                     zoomEffectiveAtCreation: Float,
                     device: MTLDevice?,
                     depthID: UInt32,
                     depthWriteEnabled: Bool = true,
                     constantScreenSize: Bool = true) {
        let safeZoom = max(zoomAtCreation, 1e-6)
        let effectiveZoom = max(zoomEffectiveAtCreation, 1e-6)

        guard let firstScreenPt = screenPoints.first else {
            self.init(
                id: UUID(),
                origin: .zero,
                worldWidth: 0,
                color: color,
                zoomEffectiveAtCreation: effectiveZoom,
                segments: [],
                localBounds: .null,
                segmentBounds: .null,
                device: device,
                depthID: depthID,
                depthWriteEnabled: depthWriteEnabled
            )
            return
        }

        // 1. CALCULATE ORIGIN (ABSOLUTE) -  HIGH PRECISION FIX
        let origin = screenToWorldPixels_PureDouble(firstScreenPt,
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

        // 3. World width calculation
        // If constantScreenSize is true, divide by zoom so stroke appears same size on screen
        // If false, scale from screen width by the effective zoom
        let worldWidth = constantScreenSize ? (baseWidth / zoom) : ((baseWidth / zoom) * Double(effectiveZoom))

        // 4. Build segment instances
        let builtSegments = Stroke.buildSegments(from: centerPoints, color: color)

        // 5. Calculate bounds expanded by stroke radius for culling
        let bounds = Stroke.calculateBounds(for: centerPoints, radius: Float(worldWidth) * 0.5)
        self.init(
            id: UUID(),
            origin: origin,
            worldWidth: worldWidth,
            color: color,
            zoomEffectiveAtCreation: effectiveZoom,
            segments: builtSegments,
            localBounds: bounds,
            segmentBounds: bounds,
            device: device,
            depthID: depthID,
            depthWriteEnabled: depthWriteEnabled
        )
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

// MARK: - Serialization
extension Stroke {
    var rawPoints: [SIMD2<Float>] {
        guard !segments.isEmpty else { return [] }

        var points: [SIMD2<Float>] = []
        points.reserveCapacity(segments.count + 1)

        for (index, segment) in segments.enumerated() {
            if index == 0 {
                points.append(segment.p0)
            }
            if let last = points.last, last != segment.p1 {
                points.append(segment.p1)
            }
        }

        return points
    }

    func toDTO() -> StrokeDTO {
        func safeDouble(_ value: Double) -> Double {
            value.isFinite ? value : 0
        }

        func safeFloat(_ value: Float, fallback: Float = 0) -> Float {
            value.isFinite ? value : fallback
        }

        let rawPts = rawPoints.compactMap { point -> [Float]? in
            guard point.x.isFinite, point.y.isFinite else { return nil }
            return [point.x, point.y]
        }

        return StrokeDTO(
            id: id,
            origin: [safeDouble(origin.x), safeDouble(origin.y)],
            worldWidth: safeDouble(worldWidth),
            color: [
                safeFloat(color.x),
                safeFloat(color.y),
                safeFloat(color.z),
                safeFloat(color.w, fallback: 1)
            ],
            zoomCreation: safeFloat(zoomEffectiveAtCreation, fallback: 1),
            depthID: depthID,
            depthWrite: depthWriteEnabled,
            points: rawPts
        )
    }

    convenience init(dto: StrokeDTO, device: MTLDevice?) {
        let localPoints = dto.points.map { point in
            let x = point.count > 0 ? point[0] : 0
            let y = point.count > 1 ? point[1] : 0
            return SIMD2<Float>(x, y)
        }

        let color = SIMD4<Float>(
            dto.color.count > 0 ? dto.color[0] : 0,
            dto.color.count > 1 ? dto.color[1] : 0,
            dto.color.count > 2 ? dto.color[2] : 0,
            dto.color.count > 3 ? dto.color[3] : 1
        )

        let segments = Stroke.buildSegments(from: localPoints, color: color)
        let bounds = Stroke.calculateBounds(for: localPoints, radius: Float(dto.worldWidth) * 0.5)

        self.init(
            id: dto.id,
            origin: SIMD2<Double>(
                dto.origin.count > 0 ? dto.origin[0] : 0,
                dto.origin.count > 1 ? dto.origin[1] : 0
            ),
            worldWidth: dto.worldWidth,
            color: color,
            zoomEffectiveAtCreation: dto.zoomCreation,
            segments: segments,
            localBounds: bounds,
            segmentBounds: bounds,
            device: device,
            depthID: dto.depthID,
            depthWriteEnabled: dto.depthWrite
        )
    }
}
