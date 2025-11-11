import SwiftUI
import Foundation
import Metal
import MetalKit
import simd

// MARK: - Geometry / Tessellation

private let baseTileSize: Double = 1024.0

struct TileKey: Hashable {
    let tx: Int64
    let ty: Int64
}

struct ViewUniforms {
    var anchorLocal: SIMD2<Float>
    var anchorNDC: SIMD2<Float>
    var tileToNDC: SIMD2<Float>
    var zoomMantissa: Float
    var zoomExponent: Int32
    var cosTheta: Float
    var sinTheta: Float
    var padding: Float = 0
}

struct TileUniforms {
    var tileDelta: SIMD2<Float>
}

final class TileGeometry {
    var vertices: [SIMD2<Float>] = []
    var buffer: MTLBuffer?

    func append(contentsOf newVertices: [SIMD2<Float>]) {
        guard !newVertices.isEmpty else { return }
        vertices.append(contentsOf: newVertices)
        buffer = nil
    }

    func makeBufferIfNeeded(device: MTLDevice) {
        guard buffer == nil, !vertices.isEmpty else { return }
        let length = vertices.count * MemoryLayout<SIMD2<Float>>.stride
        buffer = device.makeBuffer(bytes: vertices, length: length, options: .storageModeShared)
    }
}

func splitZoomScale(_ zoom: Float) -> (Int32, Float) {
    let safeZoom = max(zoom, 1e-6)
    let clampedZoom = clampZoomScale(safeZoom)

    let rawLevel = Int32(floor(log2f(clampedZoom)))
    let clampedLevel = max(min(rawLevel, maxTileLevel), minTileLevel)
    let levelScale = powf(2.0, Float(clampedLevel))
    let mantissa = clampedZoom / levelScale
    let safeMantissa = mantissa.isFinite ? mantissa : 1.0
    return (clampedLevel, safeMantissa)
}

private let maxTileLevel: Int32 = 60
private let minTileLevel: Int32 = -60
private let maxZoomScale: Float = powf(2.0, Float(maxTileLevel))
private let minZoomScale: Float = powf(2.0, Float(minTileLevel))

@inline(__always)
private func clampZoomScale(_ zoom: Float) -> Float {
    return min(max(zoom, minZoomScale), maxZoomScale)
}

func tileSize(for level: Int32) -> Double {
    let clampedLevel = max(min(level, maxTileLevel), minTileLevel)
    return baseTileSize / pow(2.0, Double(clampedLevel))
}

@inline(__always)
private func clampedInt64(_ value: Double) -> Int64 {
    guard value.isFinite else { return 0 }
    if value <= Double(Int64.min) { return Int64.min }
    if value >= Double(Int64.max) { return Int64.max }
    return Int64(value)
}

func tileKeyAndLocal(for world: SIMD2<Double>, tileSize: Double) -> (TileKey, SIMD2<Float>) {
    let key = tileKey(for: world, tileSize: tileSize)
    let local = localCoordinates(for: world, key: key, tileSize: tileSize)
    return (key, local)
}

func tileKey(for world: SIMD2<Double>, tileSize: Double) -> TileKey {
    guard tileSize.isFinite, tileSize > 0 else {
        return TileKey(tx: 0, ty: 0)
    }

    let rawX = floor(world.x / tileSize)
    let rawY = floor(world.y / tileSize)
    let tx = clampedInt64(rawX)
    let ty = clampedInt64(rawY)
    return TileKey(tx: tx, ty: ty)
}

func localCoordinates(for world: SIMD2<Double>, key: TileKey, tileSize: Double) -> SIMD2<Float> {
    guard tileSize.isFinite, tileSize > 0 else {
        return SIMD2<Float>.zero
    }

    let originX = Double(key.tx) * tileSize
    let originY = Double(key.ty) * tileSize
    let localX = Float((world.x - originX) / tileSize)
    let localY = Float((world.y - originY) / tileSize)
    if !localX.isFinite || !localY.isFinite {
        return SIMD2<Float>.zero
    }
    return SIMD2<Float>(localX, localY)
}

func tessellateStrokeWorld(centerPoints: [CGPoint], width: CGFloat) -> [SIMD2<Double>] {
    guard !centerPoints.isEmpty else { return [] }

    if centerPoints.count == 1 {
        return createCircleWorld(at: centerPoints[0], radius: width / 2.0)
    }

    var vertices: [SIMD2<Double>] = []
    let halfWidth = Double(width / 2.0)

    let startCap = createCircleWorld(at: centerPoints[0], radius: width / 2.0)
    vertices.append(contentsOf: startCap)

    for i in 0..<(centerPoints.count - 1) {
        let current = centerPoints[i]
        let next = centerPoints[i + 1]

        let p0 = SIMD2<Double>(Double(current.x), Double(current.y))
        let p1 = SIMD2<Double>(Double(next.x), Double(next.y))
        let segment = p1 - p0
        let length = simd_length(segment)
        guard length > 0 else { continue }
        let dir = segment / length
        let perp = SIMD2<Double>(-dir.y, dir.x) * halfWidth

        let t0 = p0 + perp
        let b0 = p0 - perp
        let t1 = p1 + perp
        let b1 = p1 - perp

        vertices.append(t0)
        vertices.append(b0)
        vertices.append(t1)

        vertices.append(b0)
        vertices.append(b1)
        vertices.append(t1)

        if i < centerPoints.count - 2 {
            let joint = createCircleWorld(at: next, radius: width / 2.0, segments: 16)
            vertices.append(contentsOf: joint)
        }
    }

    let endCap = createCircleWorld(at: centerPoints.last!, radius: width / 2.0)
    vertices.append(contentsOf: endCap)

    return vertices
}

func createCircleWorld(at point: CGPoint, radius: CGFloat, segments: Int = 30) -> [SIMD2<Double>] {
    guard segments >= 2 else { return [] }
    var vertices: [SIMD2<Double>] = []
    let center = SIMD2<Double>(Double(point.x), Double(point.y))
    let r = Double(radius)

    for i in 0..<segments {
        let angle1 = Double(i) * (2.0 * .pi / Double(segments))
        let angle2 = Double(i + 1) * (2.0 * .pi / Double(segments))

        let p1 = SIMD2<Double>(center.x + cos(angle1) * r,
                               center.y + sin(angle1) * r)
        let p2 = SIMD2<Double>(center.x + cos(angle2) * r,
                               center.y + sin(angle2) * r)

        vertices.append(center)
        vertices.append(p1)
        vertices.append(p2)
    }

    return vertices
}

func tessellateStrokeIntoTiles(centerPoints: [CGPoint],
                               width: CGFloat,
                               tileSize: Double) -> [TileKey: [SIMD2<Float>]] {
    let worldVertices = tessellateStrokeWorld(centerPoints: centerPoints, width: width)
    guard !worldVertices.isEmpty else { return [:] }

    var result: [TileKey: [SIMD2<Float>]] = [:]

    for i in stride(from: 0, to: worldVertices.count, by: 3) {
        let v0 = worldVertices[i]
        let v1 = worldVertices[i + 1]
        let v2 = worldVertices[i + 2]
        let centroid = (v0 + v1 + v2) / 3.0
        let key = tileKey(for: centroid, tileSize: tileSize)

        let local0 = localCoordinates(for: v0, key: key, tileSize: tileSize)
        let local1 = localCoordinates(for: v1, key: key, tileSize: tileSize)
        let local2 = localCoordinates(for: v2, key: key, tileSize: tileSize)

        result[key, default: []].append(contentsOf: [local0, local1, local2])
    }

    return result
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



// MARK: - GPU Transform Struct

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

        // Pinch anchor (persist between .began and subsequent states)
        var pinchAnchorScreen: CGPoint = .zero
        var pinchAnchorWorld: SIMD2<Float> = .zero
        var panOffsetAtPinchStart: SIMD2<Float> = .zero
        
        //roation anchor
        var rotationAnchorScreen: CGPoint = .zero
        var rotationAnchorWorld: SIMD2<Float> = .zero
        var panOffsetAtRotationStart: SIMD2<Float> = .zero
        
        enum AnchorOwner { case none, pinch, rotation }

        var activeOwner: AnchorOwner = .none
        var anchorWorld: SIMD2<Float> = .zero
        var anchorScreen: CGPoint = .zero
        
        var lastPinchTouchCount: Int = 0
        var lastRotationTouchCount: Int = 0


        
        
        func lockAnchor(owner: AnchorOwner, at screenPt: CGPoint, coord: Coordinator) {
            activeOwner = owner
            anchorScreen = screenPt
            let w = screenToWorldPixels(screenPt,
                                        viewSize: bounds.size,
                                        panOffset: coord.panOffset,
                                        zoomScale: coord.zoomScale,
                                        rotationAngle: coord.rotationAngle)
            anchorWorld = SIMD2<Float>(Float(w.x), Float(w.y))
        }

        // Re-lock anchor to a new screen point *without changing the transform*
        func relockAnchorAtCurrentCentroid(owner: AnchorOwner, screenPt: CGPoint, coord: Coordinator) {
            activeOwner = owner
            anchorScreen = screenPt
            // IMPORTANT: recompute world under the *new* centroid using current pan/zoom/rotation.
            let w = screenToWorldPixels(screenPt,
                                        viewSize: bounds.size,
                                        panOffset: coord.panOffset,
                                        zoomScale: coord.zoomScale,
                                        rotationAngle: coord.rotationAngle)
            anchorWorld = SIMD2<Float>(Float(w.x), Float(w.y))
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

            panGesture.delegate = self
            pinchGesture.delegate = self
            rotationGesture.delegate = self
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

                // Normal incremental zoom
                coord.zoomScale = clampZoomScale(coord.zoomScale * Float(gesture.scale))
                gesture.scale = 1.0

                // Keep the shared anchor pinned
                let target = (activeOwner == .pinch) ? loc : anchorScreen
                coord.panOffset = solvePanOffsetForAnchor(anchorWorld: anchorWorld,
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
            coordinator?.panOffset.x += Float(t.x)
            coordinator?.panOffset.y += Float(t.y)
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

                // Keep shared anchor pinned
                let target = (activeOwner == .rotation) ? loc : anchorScreen
                coord.panOffset = solvePanOffsetForAnchor(anchorWorld: anchorWorld,
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

    var currentTouchPoints: [CGPoint] = []
    var allStrokes: [Stroke] = []

    private var committedTileGeometry: [TileKey: TileGeometry] = [:]
    private var geometryCacheLevel: Int32?
    private var needsFullGeometryRebuild: Bool = true
    private var pendingStrokeIndices: [Int] = []

    weak var metalView: MTKView?

    var panOffset: SIMD2<Float> = .zero
    var zoomScale: Float = 1.0
    var rotationAngle: Float = 0.0

    override init() {
        super.init()
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!
        makePipeLine()
        needsFullGeometryRebuild = true
    }

    private func rebuildCommittedTileGeometry(level: Int32, tileSide: Double) {
        var newGeometry: [TileKey: TileGeometry] = [:]

        for index in allStrokes.indices {
            var stroke = allStrokes[index]
            let tiles = stroke.tiles(for: level, tileSize: tileSide)
            allStrokes[index] = stroke

            for (key, verts) in tiles {
                let geometry = newGeometry[key] ?? TileGeometry()
                geometry.append(contentsOf: verts)
                newGeometry[key] = geometry
            }
        }

        committedTileGeometry = newGeometry
        geometryCacheLevel = level
        needsFullGeometryRebuild = false
        pendingStrokeIndices.removeAll()
    }

    private func appendPendingStrokes(level: Int32, tileSide: Double) {
        guard !pendingStrokeIndices.isEmpty else { return }

        for index in pendingStrokeIndices {
            guard index < allStrokes.count else { continue }
            var stroke = allStrokes[index]
            let tiles = stroke.tiles(for: level, tileSize: tileSide)
            allStrokes[index] = stroke

            for (key, verts) in tiles {
                let geometry = committedTileGeometry[key] ?? TileGeometry()
                geometry.append(contentsOf: verts)
                committedTileGeometry[key] = geometry
            }
        }

        pendingStrokeIndices.removeAll()
    }

    func draw(in view: MTKView) {
        guard let renderPassDescriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let touchView = metalView as? MetalView.TouchableMTKView else {
            return
        }

        let (rawLevel, zoomMantissa) = splitZoomScale(zoomScale)
        let level = max(min(rawLevel, maxTileLevel), minTileLevel)
        let tileSide = tileSize(for: level)
        let viewSize = view.bounds.size

        let anchorScreen: CGPoint
        let anchorWorld: SIMD2<Float>
        if touchView.activeOwner == .none {
            let center = CGPoint(x: viewSize.width * 0.5, y: viewSize.height * 0.5)
            anchorScreen = center
            let worldPoint = screenToWorldPixels(center,
                                                viewSize: viewSize,
                                                panOffset: panOffset,
                                                zoomScale: zoomScale,
                                                rotationAngle: rotationAngle)
            anchorWorld = SIMD2<Float>(Float(worldPoint.x), Float(worldPoint.y))
        } else {
            anchorScreen = touchView.anchorScreen
            anchorWorld = touchView.anchorWorld
        }

        let anchorWorldD = SIMD2<Double>(Double(anchorWorld.x), Double(anchorWorld.y))
        let (anchorTileKey, anchorLocal) = tileKeyAndLocal(for: anchorWorldD, tileSize: tileSide)
        let anchorNDC = screenToNDC(anchorScreen, viewSize: viewSize)

        let tileToNDC = SIMD2<Float>(
            Float(tileSide / Double(viewSize.width) * 2.0),
            Float(-tileSide / Double(viewSize.height) * 2.0)
        )

        let cosTheta = Float(cos(Double(rotationAngle)))
        let sinTheta = Float(sin(Double(rotationAngle)))

        var viewUniforms = ViewUniforms(
            anchorLocal: anchorLocal,
            anchorNDC: anchorNDC,
            tileToNDC: tileToNDC,
            zoomMantissa: zoomMantissa,
            zoomExponent: level,
            cosTheta: cosTheta,
            sinTheta: sinTheta,
            padding: 0
        )

        let screenWorldCorners = [
            screenToWorldPixels(.zero,
                                 viewSize: viewSize,
                                 panOffset: panOffset,
                                 zoomScale: zoomScale,
                                 rotationAngle: rotationAngle),
            screenToWorldPixels(CGPoint(x: viewSize.width, y: 0),
                                 viewSize: viewSize,
                                 panOffset: panOffset,
                                 zoomScale: zoomScale,
                                 rotationAngle: rotationAngle),
            screenToWorldPixels(CGPoint(x: 0, y: viewSize.height),
                                 viewSize: viewSize,
                                 panOffset: panOffset,
                                 zoomScale: zoomScale,
                                 rotationAngle: rotationAngle),
            screenToWorldPixels(CGPoint(x: viewSize.width, y: viewSize.height),
                                 viewSize: viewSize,
                                 panOffset: panOffset,
                                 zoomScale: zoomScale,
                                 rotationAngle: rotationAngle)
        ]

        let xs = screenWorldCorners.map { Double($0.x) }
        let ys = screenWorldCorners.map { Double($0.y) }
        let anchorFallbackX = Double(anchorWorld.x)
        let anchorFallbackY = Double(anchorWorld.y)
        let minWorldX = xs.min() ?? anchorFallbackX
        let maxWorldX = xs.max() ?? anchorFallbackX
        let minWorldY = ys.min() ?? anchorFallbackY
        let maxWorldY = ys.max() ?? anchorFallbackY

        let margin = tileSide * 2.0

        func clampedTileIndex(_ value: Double) -> Int64 {
            guard value.isFinite else { return 0 }
            return clampedInt64(value)
        }

        let rawMinTileX = floor((minWorldX - margin) / tileSide)
        let rawMaxTileX = floor((maxWorldX + margin) / tileSide)
        let rawMinTileY = floor((minWorldY - margin) / tileSide)
        let rawMaxTileY = floor((maxWorldY + margin) / tileSide)

        let minTileX = clampedTileIndex(rawMinTileX)
        let maxTileX = clampedTileIndex(rawMaxTileX)
        let minTileY = clampedTileIndex(rawMinTileY)
        let maxTileY = clampedTileIndex(rawMaxTileY)

        let visibleTileXRange = min(minTileX, maxTileX)...max(minTileX, maxTileX)
        let visibleTileYRange = min(minTileY, maxTileY)...max(minTileY, maxTileY)

        if geometryCacheLevel != level || needsFullGeometryRebuild {
            rebuildCommittedTileGeometry(level: level, tileSide: tileSide)
        } else {
            appendPendingStrokes(level: level, tileSide: tileSide)
        }

        var dynamicTiles: [TileKey: [SIMD2<Float>]] = [:]

        if currentTouchPoints.count >= 2 {
            let strokePoints: [CGPoint]
            if currentTouchPoints.count >= 4 {
                strokePoints = catmullRomPoints(points: currentTouchPoints,
                                                closed: false,
                                                alpha: 0.5,
                                                segmentsPerCurve: 20)
            } else {
                strokePoints = currentTouchPoints
            }

            let inProgress = tessellateStrokeIntoTiles(centerPoints: strokePoints,
                                                       width: 10.0 / CGFloat(zoomScale),
                                                       tileSize: tileSide)
            for (key, verts) in inProgress {
                guard !verts.isEmpty else { continue }
                dynamicTiles[key, default: []].append(contentsOf: verts)
            }
        }

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        encoder.setRenderPipelineState(pipelineState)
        encoder.setCullMode(.none)
        encoder.setVertexBytes(&viewUniforms, length: MemoryLayout<ViewUniforms>.stride, index: 2)

        for (key, geometry) in committedTileGeometry {
            guard visibleTileXRange.contains(key.tx), visibleTileYRange.contains(key.ty) else { continue }
            guard !geometry.vertices.isEmpty else { continue }
            geometry.makeBufferIfNeeded(device: device)
            guard let buffer = geometry.buffer else { continue }
            var tileUniforms = TileUniforms(tileDelta: SIMD2<Float>(Float(key.tx - anchorTileKey.tx),
                                                                    Float(key.ty - anchorTileKey.ty)))
            encoder.setVertexBuffer(buffer, offset: 0, index: 0)
            encoder.setVertexBytes(&tileUniforms, length: MemoryLayout<TileUniforms>.stride, index: 1)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: geometry.vertices.count)
        }

        for (key, vertices) in dynamicTiles {
            guard visibleTileXRange.contains(key.tx), visibleTileYRange.contains(key.ty) else { continue }
            guard !vertices.isEmpty else { continue }
            let bufferLength = vertices.count * MemoryLayout<SIMD2<Float>>.stride
            guard let vertexBuffer = device.makeBuffer(bytes: vertices, length: bufferLength, options: .storageModeShared) else {
                continue
            }
            var tileUniforms = TileUniforms(tileDelta: SIMD2<Float>(Float(key.tx - anchorTileKey.tx),
                                                                    Float(key.ty - anchorTileKey.ty)))
            encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            encoder.setVertexBytes(&tileUniforms, length: MemoryLayout<TileUniforms>.stride, index: 1)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: vertices.count)
        }

        encoder.endEncoding()
        commandBuffer.present(drawable)
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

    // MARK: - Touch Handling

    func handleTouchBegan(at point: CGPoint) {
        guard let view = metalView else { return }
        let worldPoint = screenToWorldPixels(point,
                                             viewSize: view.bounds.size,
                                             panOffset: panOffset,
                                             zoomScale: zoomScale, rotationAngle: rotationAngle)
        currentTouchPoints = [worldPoint]
    }

    func handleTouchMoved(at point: CGPoint) {
        guard let view = metalView else { return }
        let worldPoint = screenToWorldPixels(point,
                                             viewSize: view.bounds.size,
                                             panOffset: panOffset,
                                             zoomScale: zoomScale, rotationAngle: rotationAngle)
        currentTouchPoints.append(worldPoint)
    }

    func handleTouchEnded(at point: CGPoint) {
        guard let view = metalView else { return }
        let worldPoint = screenToWorldPixels(point,
                                             viewSize: view.bounds.size,
                                             panOffset: panOffset,
                                             zoomScale: zoomScale, rotationAngle: rotationAngle)
        currentTouchPoints.append(worldPoint)

        guard currentTouchPoints.count >= 4 else {
            currentTouchPoints = []
            return
        }

        let smoothPoints = catmullRomPoints(points: currentTouchPoints,
                                            closed: false,
                                            alpha: 0.5,
                                            segmentsPerCurve: 20)

        let stroke = Stroke(centerPoints: smoothPoints,
                            width: 10.0 / CGFloat(zoomScale),
                            color: SIMD4<Float>(1.0, 0.0, 0.0, 1.0))

        allStrokes.append(stroke)
        pendingStrokeIndices.append(allStrokes.count - 1)
        currentTouchPoints = []
    }

    func handleTouchCancelled() {
        currentTouchPoints = []
    }
}

// MARK: - Gesture Delegate

extension MetalView.TouchableMTKView: UIGestureRecognizerDelegate {
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                           shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        true
    }
}
